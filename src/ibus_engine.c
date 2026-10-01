// src/ibus_engine.c — IBus front end for the Akshar engine (Rust, via c_api.rs).
//
// The Rust engine owns every language decision; this file only turns key
// events into three actions: extend the roman preedit, commit a word, commit
// a symbol.  Two invariants hold for every key:
//
//   * nothing typed is ever lost — if the engine has no answer, the raw text
//     is committed;
//   * text is committed in the order it was typed — a symbol never reaches
//     the application while a word is still pending in the preedit.
//
// IBusText ownership: ibus_text_new_from_string() returns a floating
// reference, and ibus_engine_commit_text() / ibus_engine_update_preedit_text()
// release floating text themselves.  Never unref text after handing it over.

#include <ibus.h>
#include <jansson.h>
#include <string.h>

// --- Rust FFI (src/c_api.rs) ---
void akshar_ime_engine_init(void);
void akshar_ime_engine_destroy(void);
char *akshar_ime_get_suggestions(const char *prefix);
void akshar_ime_confirm_word(const char *roman, const char *devanagari);
int akshar_ime_set_language(const char *code);
void akshar_ime_free_string(char *s);

// Engine names registered with IBus (devanagari-smart.xml): the bare name is
// "auto" (language-blind ranking); a suffix is the language the source ranks
// for, e.g. "devanagari-smart-hi".
#define ENGINE_BASE_NAME "devanagari-smart"
static const char *const LANGUAGE_SUFFIXES[] = {"hi", "ne", "mr", "sa", "kok", "mai", "brx", "doi"};

// --- GObject boilerplate ---
typedef struct _IBusDevanagariEngine IBusDevanagariEngine;
typedef struct _IBusDevanagariEngineClass IBusDevanagariEngineClass;
struct _IBusDevanagariEngine
{
    IBusEngine parent;
    IBusLookupTable *table;
    GString *preedit;
};
struct _IBusDevanagariEngineClass
{
    IBusEngineClass parent;
};

G_DEFINE_TYPE(IBusDevanagariEngine, ibus_devanagari_engine, IBUS_TYPE_ENGINE)
#define IBUS_TYPE_DEVANAGARI_ENGINE (ibus_devanagari_engine_get_type())

// One Rust engine per process, shared by every IBus engine instance.
static guint g_instances = 0;

// --- Engine calls ---

// The engine's suggestions for `input` as a JSON array; NULL on failure.
static json_t *suggestions(const char *input)
{
    char *json = akshar_ime_get_suggestions(input);
    if (!json)
        return NULL;
    json_t *root = json_loads(json, 0, NULL);
    akshar_ime_free_string(json);
    if (root && !json_is_array(root))
    {
        json_decref(root);
        return NULL;
    }
    return root;
}

// The engine's first suggestion for `input` (g_free it), or NULL.
static gchar *top_suggestion(const char *input)
{
    json_t *root = suggestions(input);
    gchar *out = NULL;
    if (root && json_array_size(root) > 0 && json_is_string(json_array_get(root, 0)))
        out = g_strdup(json_string_value(json_array_get(root, 0)));
    if (root)
        json_decref(root);
    return out;
}

static void commit_string(IBusEngine *engine, const gchar *s)
{
    ibus_engine_commit_text(engine, ibus_text_new_from_string(s));
}

// --- Preedit and lookup table ---

static void clear_preedit(IBusDevanagariEngine *self)
{
    g_string_truncate(self->preedit, 0);
    ibus_lookup_table_clear(self->table);
    ibus_engine_hide_preedit_text((IBusEngine *)self);
    ibus_engine_hide_lookup_table((IBusEngine *)self);
}

static void update_preedit_and_lookup(IBusDevanagariEngine *self)
{
    IBusEngine *engine = (IBusEngine *)self;
    if (self->preedit->len == 0)
    {
        clear_preedit(self);
        return;
    }
    const gchar *roman = self->preedit->str;
    ibus_engine_update_preedit_text(engine, ibus_text_new_from_string(roman),
                                    g_utf8_strlen(roman, -1), TRUE);

    ibus_lookup_table_clear(self->table);
    json_t *root = suggestions(roman);
    if (root)
    {
        size_t i;
        json_t *value;
        json_array_foreach(root, i, value)
        {
            if (json_is_string(value))
                ibus_lookup_table_append_candidate(
                    self->table, ibus_text_new_from_string(json_string_value(value)));
        }
        json_decref(root);
    }
    if (ibus_lookup_table_get_number_of_candidates(self->table) > 0)
        ibus_engine_update_lookup_table(engine, self->table, TRUE);
    else
        ibus_engine_hide_lookup_table(engine);
}

// Commit the highlighted candidate (the top one unless the user moved the
// cursor) and teach it to the engine.  With no candidate at all the typed
// roman is committed as-is rather than dropped.
static void commit_preedit(IBusDevanagariEngine *self)
{
    if (self->preedit->len == 0)
        return;
    gchar *roman = g_strdup(self->preedit->str);
    gchar *word = NULL;

    guint n = ibus_lookup_table_get_number_of_candidates(self->table);
    guint cursor = ibus_lookup_table_get_cursor_pos(self->table);
    if (cursor < n)
    {
        // Owned by the table (transfer none): copy the string, keep the object.
        IBusText *cand = ibus_lookup_table_get_candidate(self->table, cursor);
        if (cand && cand->text)
            word = g_strdup(cand->text);
    }
    if (!word)
        word = top_suggestion(roman);

    commit_string((IBusEngine *)self, word ? word : roman);
    if (word)
        akshar_ime_confirm_word(roman, word);
    g_free(word);
    g_free(roman);
    clear_preedit(self);
}

// --- Key handling ---

static gboolean ibus_devanagari_engine_process_key_event(IBusEngine *engine, guint keyval,
                                                         guint keycode, guint modifiers)
{
    IBusDevanagariEngine *self = (IBusDevanagariEngine *)engine;
    (void)keycode;

    if (modifiers & IBUS_RELEASE_MASK)
        return FALSE;
    // Application shortcuts (Ctrl/Alt/Super + key) are not text: let them
    // through untouched.
    if (modifiers & (IBUS_CONTROL_MASK | IBUS_MOD1_MASK | IBUS_SUPER_MASK))
        return FALSE;

    gboolean has_preedit = self->preedit->len > 0;

    if (has_preedit && ibus_lookup_table_get_number_of_candidates(self->table) > 0)
    {
        switch (keyval)
        {
        case IBUS_KEY_Up:
            ibus_lookup_table_cursor_up(self->table);
            ibus_engine_update_lookup_table(engine, self->table, TRUE);
            return TRUE;
        case IBUS_KEY_Down:
            ibus_lookup_table_cursor_down(self->table);
            ibus_engine_update_lookup_table(engine, self->table, TRUE);
            return TRUE;
        }
    }

    switch (keyval)
    {
    case IBUS_KEY_Return:
    case IBUS_KEY_KP_Enter:
    case IBUS_KEY_Tab:
        // With a word pending the key commits the highlighted candidate.
        if (!has_preedit)
            return FALSE;
        commit_preedit(self);
        return TRUE;
    case IBUS_KEY_space:
        // Space commits the highlighted candidate AND advances with a space.
        if (!has_preedit)
            return FALSE;
        commit_preedit(self);
        commit_string(engine, " ");
        return TRUE;
    case IBUS_KEY_Escape:
        if (!has_preedit)
            return FALSE;
        clear_preedit(self);
        return TRUE;
    case IBUS_KEY_BackSpace:
        if (!has_preedit)
            return FALSE;
        // The preedit is ASCII, but step back a whole UTF-8 character anyway.
        g_string_truncate(self->preedit,
                          g_utf8_prev_char(self->preedit->str + self->preedit->len) -
                              self->preedit->str);
        update_preedit_and_lookup(self);
        return TRUE;
    }

    // Direct candidate selection with number keys 1-9 while preedit is active.
    if (has_preedit && keyval >= '1' && keyval <= '9')
    {
        guint idx = keyval - '1';
        guint n = ibus_lookup_table_get_number_of_candidates(self->table);
        if (idx < n)
        {
            ibus_lookup_table_set_cursor_pos(self->table, idx);
            commit_preedit(self);
            return TRUE;
        }
    }

    // Letters build the roman word.
    if (keyval < 0x80 && g_ascii_isalpha((gchar)keyval))
    {
        g_string_append_c(self->preedit, (gchar)keyval);
        update_preedit_and_lookup(self);
        return TRUE;
    }

    // Digits and punctuation: commit the pending word first, then the symbol
    // as the engine maps it ('.' -> '।', digits -> Devanagari digits, the
    // rest unchanged), falling back to the raw character.
    if (keyval >= 0x21 && keyval <= 0x7e)
    {
        if (has_preedit)
            commit_preedit(self);
        gchar symbol[2] = {(gchar)keyval, '\0'};
        gchar *mapped = top_suggestion(symbol);
        commit_string(engine, mapped ? mapped : symbol);
        g_free(mapped);
        return TRUE;
    }

    // Cursor movement and editing keys belong to the application, but must
    // not act while a word is pending in front of the cursor.  (Modifier keys
    // such as Shift arrive as their own events and fall through untouched.)
    switch (keyval)
    {
    case IBUS_KEY_Left:
    case IBUS_KEY_Right:
    case IBUS_KEY_Home:
    case IBUS_KEY_End:
    case IBUS_KEY_Page_Up:
    case IBUS_KEY_Page_Down:
    case IBUS_KEY_Delete:
    case IBUS_KEY_Insert:
        if (has_preedit)
            commit_preedit(self);
        break;
    }
    return FALSE;
}

static void ibus_devanagari_engine_candidate_clicked(IBusEngine *engine, guint index,
                                                     guint button, guint state)
{
    IBusDevanagariEngine *self = (IBusDevanagariEngine *)engine;
    (void)button;
    (void)state;
    guint page_start = ibus_lookup_table_get_cursor_pos(self->table) -
                       ibus_lookup_table_get_cursor_in_page(self->table);
    ibus_lookup_table_set_cursor_pos(self->table, page_start + index);
    commit_preedit(self);
}

// Every input source shares one Rust engine, so each tells it its language
// whenever it becomes the active source.
static void apply_language(IBusEngine *engine)
{
    const gchar *name = ibus_engine_get_name(engine);
    const gchar *lang = NULL;
    if (name && g_str_has_prefix(name, ENGINE_BASE_NAME "-"))
        lang = name + strlen(ENGINE_BASE_NAME "-");
    akshar_ime_set_language(lang);
}

static void ibus_devanagari_engine_focus_in(IBusEngine *engine)
{
    apply_language(engine);
    IBUS_ENGINE_CLASS(ibus_devanagari_engine_parent_class)->focus_in(engine);
}

static void ibus_devanagari_engine_enable(IBusEngine *engine)
{
    apply_language(engine);
    IBUS_ENGINE_CLASS(ibus_devanagari_engine_parent_class)->enable(engine);
}

// Focus leaving or the engine being switched off must not lose the word.
static void ibus_devanagari_engine_focus_out(IBusEngine *engine)
{
    commit_preedit((IBusDevanagariEngine *)engine);
    IBUS_ENGINE_CLASS(ibus_devanagari_engine_parent_class)->focus_out(engine);
}

static void ibus_devanagari_engine_disable(IBusEngine *engine)
{
    commit_preedit((IBusDevanagariEngine *)engine);
    IBUS_ENGINE_CLASS(ibus_devanagari_engine_parent_class)->disable(engine);
}

// The client asked for a reset (e.g. its text was replaced): drop the preedit.
static void ibus_devanagari_engine_reset(IBusEngine *engine)
{
    clear_preedit((IBusDevanagariEngine *)engine);
    IBUS_ENGINE_CLASS(ibus_devanagari_engine_parent_class)->reset(engine);
}

// --- Lifecycle ---

static void ibus_devanagari_engine_init(IBusDevanagariEngine *self)
{
    self->preedit = g_string_new("");
    self->table = ibus_lookup_table_new(10, 0, TRUE, TRUE);
    g_object_ref_sink(self->table);
    if (g_instances++ == 0)
        akshar_ime_engine_init();
}

static void ibus_devanagari_engine_finalize(GObject *object)
{
    IBusDevanagariEngine *self = (IBusDevanagariEngine *)object;
    g_clear_object(&self->table);
    if (self->preedit)
    {
        g_string_free(self->preedit, TRUE);
        self->preedit = NULL;
    }
    if (--g_instances == 0)
        akshar_ime_engine_destroy();
    G_OBJECT_CLASS(ibus_devanagari_engine_parent_class)->finalize(object);
}

static void ibus_devanagari_engine_class_init(IBusDevanagariEngineClass *klass)
{
    IBusEngineClass *engine_class = IBUS_ENGINE_CLASS(klass);
    engine_class->process_key_event = ibus_devanagari_engine_process_key_event;
    engine_class->candidate_clicked = ibus_devanagari_engine_candidate_clicked;
    engine_class->focus_in = ibus_devanagari_engine_focus_in;
    engine_class->enable = ibus_devanagari_engine_enable;
    engine_class->focus_out = ibus_devanagari_engine_focus_out;
    engine_class->disable = ibus_devanagari_engine_disable;
    engine_class->reset = ibus_devanagari_engine_reset;
    G_OBJECT_CLASS(klass)->finalize = ibus_devanagari_engine_finalize;
}

int main(int argc, char **argv)
{
    ibus_init();
    IBusBus *bus = ibus_bus_new();
    if (!ibus_bus_is_connected(bus))
    {
        g_object_unref(bus);
        return 1;
    }
    IBusFactory *factory = ibus_factory_new(ibus_bus_get_connection(bus));
    ibus_factory_add_engine(factory, ENGINE_BASE_NAME, IBUS_TYPE_DEVANAGARI_ENGINE);
    for (gsize i = 0; i < G_N_ELEMENTS(LANGUAGE_SUFFIXES); i++)
    {
        gchar *name = g_strconcat(ENGINE_BASE_NAME "-", LANGUAGE_SUFFIXES[i], NULL);
        ibus_factory_add_engine(factory, name, IBUS_TYPE_DEVANAGARI_ENGINE);
        g_free(name);
    }
    if (argc > 1 && strcmp(argv[1], "--ibus") == 0 &&
        !ibus_bus_request_name(bus, "org.freedesktop.IBus.AksharDevanagari", 0))
    {
        g_object_unref(factory);
        g_object_unref(bus);
        return 1;
    }
    ibus_main();
    g_object_unref(factory);
    g_object_unref(bus);
    return 0;
}
