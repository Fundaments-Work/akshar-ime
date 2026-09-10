// src/ibus_engine.c
#include <ibus.h>
#include <jansson.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

// --- Rust FFI function declarations ---
void akshar_ime_engine_init(void);
void akshar_ime_engine_destroy(void);
char *akshar_ime_get_suggestions(const char *prefix);
void akshar_ime_confirm_word(const char *roman, const char *devanagari);
void akshar_ime_free_string(char *s);

// --- GObject Boilerplate ---
typedef struct _IBusDevanagariEngine IBusDevanagariEngine;
typedef struct _IBusDevanagariEngineClass IBusDevanagariEngineClass;
struct _IBusDevanagariEngine
{
    IBusEngine parent;
    IBusLookupTable *table;
    GString *preedit_string;
};
struct _IBusDevanagariEngineClass
{
    IBusEngineClass parent;
};
static guint g_engine_instance_count = 0;
static void ibus_devanagari_engine_class_init(IBusDevanagariEngineClass *klass);
static void ibus_devanagari_engine_init_instance(IBusDevanagariEngine *engine);
static void ibus_devanagari_engine_finalize(GObject *object);
static gboolean ibus_devanagari_engine_process_key_event(IBusEngine *engine, guint keyval, guint keycode, guint modifiers);
static void ibus_devanagari_engine_candidate_clicked(IBusEngine *engine, guint index, guint button, guint state);
G_DEFINE_TYPE(IBusDevanagariEngine, ibus_devanagari_engine, IBUS_TYPE_ENGINE)
#define IBUS_TYPE_DEVANAGARI_ENGINE (ibus_devanagari_engine_get_type())

// --- Initialization and Finalization ---
static void ibus_devanagari_engine_class_init(IBusDevanagariEngineClass *klass)
{
    IBusEngineClass *engine_class = IBUS_ENGINE_CLASS(klass);
    GObjectClass *object_class = G_OBJECT_CLASS(klass);
    engine_class->process_key_event = ibus_devanagari_engine_process_key_event;
    engine_class->candidate_clicked = ibus_devanagari_engine_candidate_clicked;
    object_class->finalize = ibus_devanagari_engine_finalize;
}
static void ibus_devanagari_engine_init_instance(IBusDevanagariEngine *engine)
{
    engine->preedit_string = g_string_new("");
    engine->table = ibus_lookup_table_new(10, 0, TRUE, TRUE);
    g_object_ref_sink(engine->table);
    if (g_engine_instance_count == 0)
    {
        akshar_ime_engine_init();
    }
    g_engine_instance_count++;
}
static void ibus_devanagari_engine_init(IBusDevanagariEngine *engine) { ibus_devanagari_engine_init_instance(engine); }
static void ibus_devanagari_engine_finalize(GObject *object)
{
    IBusDevanagariEngine *self = (IBusDevanagariEngine *)object;
    // Exp 4: previously leaked preedit_string + table on every engine switch.
    if (self->table) {
        g_object_unref(self->table);
        self->table = NULL;
    }
    if (self->preedit_string) {
        g_string_free(self->preedit_string, TRUE);
        self->preedit_string = NULL;
    }
    g_engine_instance_count--;
    if (g_engine_instance_count == 0)
    {
        akshar_ime_engine_destroy();
    }
    G_OBJECT_CLASS(ibus_devanagari_engine_parent_class)->finalize(object);
}

// --- Core IME Logic ---
static void clear_preedit(IBusDevanagariEngine *devanagari_engine)
{
    g_string_set_size(devanagari_engine->preedit_string, 0);
    ibus_engine_hide_preedit_text((IBusEngine *)devanagari_engine);
    ibus_engine_hide_lookup_table((IBusEngine *)devanagari_engine);
}

static void update_preedit_and_lookup(IBusDevanagariEngine *devanagari_engine)
{
    IBusEngine *engine = (IBusEngine *)devanagari_engine;
    const char *preedit_str = devanagari_engine->preedit_string->str;

    if (strlen(preedit_str) == 0)
    {
    clear_preedit(devanagari_engine);
        return;
    }

    IBusText *preedit_text = ibus_text_new_from_string(preedit_str);
    ibus_engine_update_preedit_text(engine, preedit_text, g_utf8_strlen(preedit_str, -1), TRUE);
    ibus_lookup_table_clear(devanagari_engine->table);

    char *suggestions_json = akshar_ime_get_suggestions(preedit_str);
    if (!suggestions_json)
        return;
    json_error_t error;
    json_t *root = json_loads(suggestions_json, 0, &error);

    if (root && json_is_array(root))
    {
        size_t i;
        json_t *value;
        json_array_foreach(root, i, value)
        {
            if (json_is_string(value))
            {
                IBusText *candidate_text = ibus_text_new_from_string(json_string_value(value));
                ibus_lookup_table_append_candidate(devanagari_engine->table, candidate_text);
            }
        }
        json_decref(root);
    }
    akshar_ime_free_string(suggestions_json);

    if (ibus_lookup_table_get_number_of_candidates(devanagari_engine->table) > 0)
    {
        ibus_engine_update_lookup_table(engine, devanagari_engine->table, TRUE);
    }
    else
    {
        ibus_engine_hide_lookup_table(engine);
    }
}

// Commits the currently selected candidate or the top suggestion if none is selected.
static void commit_best_candidate(IBusDevanagariEngine *devanagari_engine)
{
    if (devanagari_engine->preedit_string->len == 0)
        return;

    const char *preedit_for_confirm = g_strdup(devanagari_engine->preedit_string->str);
    IBusText *commit_text = NULL;

    // First, try to get the user-selected candidate (bounds-checked: the
    // table may have been repopulated since the cursor moved).
    guint n_candidates = ibus_lookup_table_get_number_of_candidates(devanagari_engine->table);
    guint index = ibus_lookup_table_get_cursor_pos(devanagari_engine->table);
    if (index < n_candidates)
        commit_text = ibus_lookup_table_get_candidate(devanagari_engine->table, index);
    if (commit_text)
    {
        g_object_ref(commit_text); // Increment ref count because we are using it
    }

    // If no candidate is selected, fetch the top suggestion directly from Rust
    if (!commit_text)
    {
    char *suggestions_json = akshar_ime_get_suggestions(preedit_for_confirm);
        if (!suggestions_json)
        {
            g_free((gpointer)preedit_for_confirm);
            return;
        }
        json_error_t error;
        json_t *root = json_loads(suggestions_json, 0, &error);
        if (root && json_is_array(root) && json_array_size(root) > 0)
        {
            json_t *first = json_array_get(root, 0);
            if (json_is_string(first))
            {
                commit_text = ibus_text_new_from_string(json_string_value(first));
            }
        }
        if (root)
            json_decref(root);
    akshar_ime_free_string(suggestions_json);
    }

    if (commit_text && commit_text->text)
    {
    ibus_engine_commit_text((IBusEngine *)devanagari_engine, commit_text);
    akshar_ime_confirm_word(preedit_for_confirm, commit_text->text);
    }

    if (commit_text)
    {
        g_object_unref(commit_text);
    }
    g_free((gpointer)preedit_for_confirm);
    clear_preedit(devanagari_engine);
}

static void ibus_devanagari_engine_candidate_clicked(IBusEngine *engine, guint index, guint button, guint state)
{
    ibus_lookup_table_set_cursor_pos(((IBusDevanagariEngine *)engine)->table, index);
    commit_best_candidate((IBusDevanagariEngine *)engine);
}

// --- RE-ARCHITECTED: The main key event processor ---
static gboolean ibus_devanagari_engine_process_key_event(IBusEngine *engine, guint keyval, guint keycode, guint modifiers)
{
    IBusDevanagariEngine *devanagari_engine = (IBusDevanagariEngine *)engine;

    if ((modifiers & IBUS_RELEASE_MASK) || (modifiers & (IBUS_CONTROL_MASK | IBUS_MOD1_MASK)))
    {
        return FALSE;
    }

    gboolean has_preedit = (devanagari_engine->preedit_string->len > 0);
    gboolean has_candidates = ibus_lookup_table_get_number_of_candidates(devanagari_engine->table) > 0;

    // --- Punctuation and Symbol Handling ---
    // Check for symbols that should immediately commit.
    if (keyval == IBUS_KEY_period || keyval == IBUS_KEY_question || keyval == IBUS_KEY_comma ||
        (keyval >= IBUS_KEY_0 && keyval <= IBUS_KEY_9))
    {
        if (has_preedit)
        {
            commit_best_candidate(devanagari_engine);
        }
        // Now, transliterate and commit the symbol itself
        char symbol_str[2] = {(char)keyval, '\0'};
        char *suggestions_json = akshar_ime_get_suggestions(symbol_str);
        if (!suggestions_json)
            return TRUE;
        json_error_t error;
        json_t *root = json_loads(suggestions_json, 0, &error);
        if (root && json_is_array(root) && json_array_size(root) > 0)
        {
            json_t *first = json_array_get(root, 0);
            if (json_is_string(first))
            {
                IBusText *text = ibus_text_new_from_string(json_string_value(first));
                ibus_engine_commit_text(engine, text);
                g_object_unref(text);
            }
        }
        if (root)
            json_decref(root);
    akshar_ime_free_string(suggestions_json);
        return TRUE; // Consume the key event
    }

    // --- Candidate Navigation ---
    if (has_candidates)
    {
        switch (keyval)
        {
        case IBUS_KEY_Up:
            ibus_lookup_table_cursor_up(devanagari_engine->table);
            ibus_engine_update_lookup_table(engine, devanagari_engine->table, TRUE);
            return TRUE;
        case IBUS_KEY_Down:
            ibus_lookup_table_cursor_down(devanagari_engine->table);
            ibus_engine_update_lookup_table(engine, devanagari_engine->table, TRUE);
            return TRUE;
        }
    }

    // --- Keypress Processing ---
    switch (keyval)
    {
    case IBUS_KEY_Return:
    case IBUS_KEY_space:
    case IBUS_KEY_Tab:
        if (has_preedit)
        {
            commit_best_candidate(devanagari_engine);
            return TRUE; // Consume the event to prevent extra space/enter.
        }
        return FALSE; // No preedit, so pass the key to the application.

    case IBUS_KEY_Escape:
        if (has_preedit)
        {
            clear_preedit(devanagari_engine);
            return TRUE;
        }
        return FALSE;

    case IBUS_KEY_BackSpace:
        if (has_preedit)
        {
            // Exp 4: byte-truncate splits UTF-8. Step back one full character.
            GString *s = devanagari_engine->preedit_string;
            if (s->len > 0) {
                const gchar *end = s->str + s->len;
                const gchar *prev = g_utf8_prev_char(end);
                g_string_truncate(s, prev - s->str);
            }
            update_preedit_and_lookup(devanagari_engine);
            return TRUE;
        }
        return FALSE;
    }

    // --- Alphanumeric Input ---
    if ((keyval >= IBUS_KEY_a && keyval <= IBUS_KEY_z) || (keyval >= IBUS_KEY_A && keyval <= IBUS_KEY_Z) || (keyval >= IBUS_KEY_0 && keyval <= IBUS_KEY_9))
    {
    g_string_append_c(devanagari_engine->preedit_string, (gchar)keyval);
    update_preedit_and_lookup(devanagari_engine);
    return TRUE;
    }

    return FALSE;
}

// --- Main Function (unchanged) ---
int main(int argc, char **argv)
{
    ibus_init();
    IBusBus *bus = ibus_bus_new();
    if (!ibus_bus_is_connected(bus))
    {
        return 1;
    }
    IBusFactory *factory = ibus_factory_new(ibus_bus_get_connection(bus));
    ibus_factory_add_engine(factory, "devanagari-smart", IBUS_TYPE_DEVANAGARI_ENGINE);
    if (argc > 1 && strcmp(argv[1], "--ibus") == 0)
    {
        if (!ibus_bus_request_name(bus, "org.freedesktop.IBus.AksharDevanagari", 0)) {
            g_object_unref(factory);
            g_object_unref(bus);
            return 1;
        }
    }
    ibus_main();
    g_object_unref(factory);
    g_object_unref(bus);
    return 0;
}