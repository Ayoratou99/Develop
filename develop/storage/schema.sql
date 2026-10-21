-- Schéma de la « mémoire Develop » (SQLite).
--
-- Appliqué de façon idempotente par Database.migrate() : chaque instruction utilise
-- IF NOT EXISTS, le fichier peut donc être rejoué sans risque. Les ids entiers sont
-- AUTOINCREMENT : un id supprimé n'est jamais réattribué (clés de noeud stables).
-- Les PRAGMA de connexion (journal_mode, foreign_keys, synchronous) sont aussi
-- positionnés par db.py à l'ouverture, car ils sont propres à chaque connexion.
--
-- Conventions :
--   * colonnes "JSON" (meta, fields, relations, tags, sources, stats) : texte JSON ;
--   * dates : REAL = secondes epoch (time.time()), sauf notes.note_date 'YYYY-MM-DD' ;
--   * noeuds du graphe (relations.src / dst) : clés textuelles
--     "file:<id>", "symbol:<id>", "endpoint:<id>", "model:<id>",
--     "document:<id>", "note:<id>", "table:<nom>".

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
PRAGMA synchronous = NORMAL;

-- ---------------------------------------------------------------- métadonnées
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

-- ---------------------------------------------------------------- dépôts / fichiers
CREATE TABLE IF NOT EXISTS repos (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL DEFAULT '',
    path            TEXT NOT NULL DEFAULT '',
    kind            TEXT NOT NULL DEFAULT 'other',
    head_sha        TEXT,
    last_indexed_at REAL
);

CREATE TABLE IF NOT EXISTS files (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_id    TEXT NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
    path       TEXT NOT NULL,
    language   TEXT,
    kind       TEXT,
    size       INTEGER,
    mtime      REAL,
    sha1       TEXT,
    indexed_at REAL,
    UNIQUE (repo_id, path)
);
CREATE INDEX IF NOT EXISTS idx_files_sha1 ON files(sha1);

-- ---------------------------------------------------------------- artefacts d'analyse
CREATE TABLE IF NOT EXISTS symbols (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id    INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    name       TEXT NOT NULL,
    kind       TEXT,
    line_start INTEGER,
    line_end   INTEGER,
    signature  TEXT,
    parent     TEXT,
    meta       TEXT
);
CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file_id);

CREATE TABLE IF NOT EXISTS endpoints (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id         INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    side            TEXT NOT NULL CHECK (side IN ('server', 'client')),
    method          TEXT,
    path            TEXT,
    normalized_path TEXT,
    handler         TEXT,
    framework       TEXT,
    line            INTEGER,
    meta            TEXT
);
CREATE INDEX IF NOT EXISTS idx_endpoints_normalized_path ON endpoints(normalized_path);
CREATE INDEX IF NOT EXISTS idx_endpoints_file ON endpoints(file_id);

CREATE TABLE IF NOT EXISTS data_models (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id    INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    name       TEXT NOT NULL,
    table_name TEXT,
    kind       TEXT,
    fields     TEXT,
    relations  TEXT,
    line       INTEGER
);
CREATE INDEX IF NOT EXISTS idx_data_models_name ON data_models(name);
CREATE INDEX IF NOT EXISTS idx_data_models_table ON data_models(table_name);
CREATE INDEX IF NOT EXISTS idx_data_models_file ON data_models(file_id);

-- ---------------------------------------------------------------- graphe de relations
CREATE TABLE IF NOT EXISTS relations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    src        TEXT NOT NULL,
    dst        TEXT NOT NULL,
    kind       TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    meta       TEXT,
    UNIQUE (src, dst, kind)
);
CREATE INDEX IF NOT EXISTS idx_relations_src ON relations(src);
CREATE INDEX IF NOT EXISTS idx_relations_dst ON relations(dst);

-- ---------------------------------------------------------------- documents / notes
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL DEFAULT '',
    source_path TEXT,
    stored_path TEXT,
    kind        TEXT NOT NULL DEFAULT 'upload' CHECK (kind IN ('upload', 'repo_file')),
    format      TEXT,
    size        INTEGER,
    sha1        TEXT,
    added_at    REAL,
    updated_at  REAL,
    meta        TEXT
);
CREATE INDEX IF NOT EXISTS idx_documents_sha1 ON documents(sha1);

CREATE TABLE IF NOT EXISTS notes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    title      TEXT NOT NULL DEFAULT '',
    body       TEXT NOT NULL DEFAULT '',
    tags       TEXT,
    note_date  TEXT,
    created_at REAL,
    updated_at REAL
);
CREATE INDEX IF NOT EXISTS idx_notes_date ON notes(note_date);

-- ---------------------------------------------------------------- chunks + recherche lexicale
CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type TEXT NOT NULL CHECK (source_type IN ('file', 'document', 'note')),
    source_id   INTEGER NOT NULL,
    ordinal     INTEGER NOT NULL,
    text        TEXT NOT NULL,
    line_start  INTEGER,
    line_end    INTEGER,
    sha1        TEXT,
    meta        TEXT,
    UNIQUE (source_type, source_id, ordinal)
);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_type, source_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content = 'chunks',
    content_rowid = 'id',
    tokenize = 'unicode61 remove_diacritics 2'
);

-- Synchronisation chunks -> chunks_fts (table de contenu externe).
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;

-- Filet de sécurité : chunks.source_id n'est pas une clé étrangère (polymorphe), on
-- supprime donc les chunks d'une source quand la source disparaît.
CREATE TRIGGER IF NOT EXISTS files_ad_chunks AFTER DELETE ON files BEGIN
    DELETE FROM chunks WHERE source_type = 'file' AND source_id = old.id;
END;
CREATE TRIGGER IF NOT EXISTS documents_ad_chunks AFTER DELETE ON documents BEGIN
    DELETE FROM chunks WHERE source_type = 'document' AND source_id = old.id;
END;
CREATE TRIGGER IF NOT EXISTS notes_ad_chunks AFTER DELETE ON notes BEGIN
    DELETE FROM chunks WHERE source_type = 'note' AND source_id = old.id;
END;

-- ---------------------------------------------------------------- embeddings
CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    model    TEXT NOT NULL,
    dim      INTEGER NOT NULL,
    vector   BLOB NOT NULL            -- float32 little-endian, dim valeurs
);
CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model);

-- ---------------------------------------------------------------- conversations
CREATE TABLE IF NOT EXISTS conversations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    title      TEXT NOT NULL DEFAULT '',
    created_at REAL,
    updated_at REAL
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL DEFAULT '',
    sources         TEXT,
    created_at      REAL
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);

-- ---------------------------------------------------------------- journaux d'indexation
CREATE TABLE IF NOT EXISTS index_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  REAL,
    finished_at REAL,
    status      TEXT,
    stats       TEXT,
    error       TEXT
);
