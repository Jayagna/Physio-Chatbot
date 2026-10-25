-- PhysioBOT knowledge base, stored in the same Supabase project as the
-- clinic website. Run once in the Supabase SQL editor.

create extension if not exists vector;

create table if not exists public.documents (
  id bigint generated always as identity primary key,
  source text not null,          -- e.g. "Tidy's Physiotherapy"
  page int,                      -- 1-based page number in the source PDF
  chunk_index int not null,
  content text not null,
  embedding vector(768) not null,
  created_at timestamptz not null default now(),
  unique (source, chunk_index)
);

create index if not exists documents_embedding_idx
  on public.documents using hnsw (embedding vector_cosine_ops);

-- Only the server (service role key) touches this table.
alter table public.documents enable row level security;

create or replace function public.match_documents(
  query_embedding vector(768),
  match_count int default 6,
  min_similarity float default 0.5
)
returns table (id bigint, source text, page int, content text, similarity float)
language sql
stable
set search_path = public
as $$
  select d.id, d.source, d.page, d.content, 1 - (d.embedding <=> query_embedding) as similarity
  from public.documents d
  where 1 - (d.embedding <=> query_embedding) >= min_similarity
  order by d.embedding <=> query_embedding
  limit least(match_count, 20);
$$;

revoke execute on function public.match_documents(vector, int, float) from public, anon, authenticated;

-- Also defined by the website's migration; harmless to re-create.
create or replace function public.ping()
returns text language sql stable as $$ select 'ok'::text $$;
