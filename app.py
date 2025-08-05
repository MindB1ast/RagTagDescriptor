import streamlit as st
import json
import os
import uuid
import time
from dotenv import load_dotenv
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
import google.generativeai as genai

# --- Загрузка API ключа ---
load_dotenv()
genai.configure(api_key=os.getenv("GEMENI_API_KEY"))

# --- Настройки ChromaDB ---
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "tag_docs"

# --- Функция эмбендинга ---
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        model = 'models/embedding-001'
        title = "tags"
        return genai.embed_content(
            model=model,
            content=input,
            task_type="retrieval_document",
            title=title
        )["embedding"]

# --- Работа с ChromaDB ---
def get_chroma_db():
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    return chroma_client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=GeminiEmbeddingFunction())

# --- Форматирование строки для эмбендинга ---
def format_entry(tag_dict):
    return f"""Tag: {tag_dict['tag']}
Description: {tag_dict['description']}
Instruction: {tag_dict.get('merge_instruction', '')}
Category: {tag_dict.get('category', '')}
"""

# --- Управление синхронизацией с ChromaDB ---
def get_tag_id(tag_name):
    """Генерирует стабильный ID для тега на основе его имени"""
    return f"tag_{hash(tag_name) % (10**8)}"

def sync_tag_to_chroma(tag_data, db):
    """Добавляет или обновляет тег в ChromaDB"""
    tag_id = get_tag_id(tag_data['tag'])
    doc_str = format_entry(tag_data)
    
    try:
        # Проверяем, существует ли уже этот тег
        existing = db.get(ids=[tag_id])
        if existing['ids']:
            # Обновляем существующий
            db.update(ids=[tag_id], documents=[doc_str])
            return f"Updated tag '{tag_data['tag']}' in ChromaDB"
        else:
            # Добавляем новый
            db.add(documents=[doc_str], ids=[tag_id])
            return f"Added tag '{tag_data['tag']}' to ChromaDB"
    except Exception as e:
        return f"Error syncing tag '{tag_data['tag']}': {str(e)}"

def remove_tag_from_chroma(tag_name, db):
    """Удаляет тег из ChromaDB"""
    tag_id = get_tag_id(tag_name)
    try:
        db.delete(ids=[tag_id])
        return f"Removed tag '{tag_name}' from ChromaDB"
    except Exception as e:
        return f"Error removing tag '{tag_name}': {str(e)}"

def get_chroma_status():
    """Получает информацию о том, какие теги есть в ChromaDB"""
    try:
        db = get_chroma_db()
        all_items = db.get()
        chroma_ids = set(all_items['ids'])
        return chroma_ids, db
    except Exception as e:
        st.error(f"Error accessing ChromaDB: {e}")
        return set(), None

def check_sync_status(tags_data, chroma_ids):
    """Проверяет статус синхронизации тегов с ChromaDB"""
    sync_status = {}
    for tag in tags_data:
        tag_id = get_tag_id(tag['tag'])
        sync_status[tag['tag']] = tag_id in chroma_ids
    return sync_status

# --- Работа с JSON ---
JSON_PATH = "tag_docs.json"

def load_tags():
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_tags(data):
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

tags_data = load_tags()

# --- Интерфейс Streamlit ---
st.set_page_config(page_title="Tag Documentation", layout="wide")
st.title("🧷 Tag Documentation Editor")

# --- Получаем статус ChromaDB ---
chroma_ids, db = get_chroma_status()
sync_status = check_sync_status(tags_data, chroma_ids) if db else {}

# --- Колонки для основного интерфейса ---
col1, col2 = st.columns([2, 1])

with col1:
    # --- Поиск и выбор ---
    search_query = st.text_input("Search", "")
    filtered_tags = [tag for tag in tags_data if search_query.lower() in tag["tag"].lower()]
    existing_tag_names = [t["tag"] for t in filtered_tags]

    selected_tag = st.selectbox("Choose existing tag to edit", [""] + existing_tag_names)
    new_tag_input = st.text_input("Or enter a new tag to create", "")

    # Определяем текущий тег
    if new_tag_input.strip():
        current_tag = new_tag_input.strip()
        tag_data = {"tag": current_tag, "description": "", "merge_instruction": "", "category": ""}
    elif selected_tag:
        current_tag = selected_tag
        tag_data = next((t for t in tags_data if t["tag"] == selected_tag), {"tag": "", "description": "", "merge_instruction": "", "category": ""})
    else:
        current_tag = ""
        tag_data = {"tag": "", "description": "", "merge_instruction": "", "category": ""}

    # --- Форма для создания/редактирования ---
    with st.form("tag_form"):
        st.text_input("Tag", value=tag_data.get("tag", ""), disabled=True)
        description = st.text_area("Description", tag_data.get("description", ""))
        merge_instruction = st.text_area("Merge Instruction (optional)", tag_data.get("merge_instruction", ""))
        category = st.text_input("Category (optional)", tag_data.get("category", ""))

        # Кнопки сохранения и удаления
        col_save, col_delete = st.columns(2)
        
        with col_save:
            submitted = st.form_submit_button("💾 Save", use_container_width=True)
        
        with col_delete:
            delete_clicked = st.form_submit_button("🗑️ Delete", use_container_width=True, type="secondary")

        # Обработка сохранения
        if submitted and current_tag:
            new_entry = {
                "tag": current_tag,
                "description": description.strip(),
                "merge_instruction": merge_instruction.strip(),
                "category": category.strip()
            }

            # Обновляем JSON
            tags_data = [t for t in tags_data if t["tag"] != current_tag]
            tags_data.append(new_entry)
            save_tags(tags_data)

            # Обновляем ChromaDB
            if db:
                try:
                    result = sync_tag_to_chroma(new_entry, db)
                    st.success(f"✅ {result}")
                    time.sleep(0.5)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to sync with ChromaDB: {e}")
            else:
                st.warning("⚠️ Saved to JSON, but ChromaDB is not available")

        # Обработка удаления
        if delete_clicked and current_tag and selected_tag:
            # Удаляем из JSON
            tags_data = [t for t in tags_data if t["tag"] != current_tag]
            save_tags(tags_data)
            
            # Удаляем из ChromaDB
            if db:
                try:
                    result = remove_tag_from_chroma(current_tag, db)
                    st.success(f"✅ Deleted '{current_tag}' from both JSON and ChromaDB")
                    time.sleep(0.5)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Deleted from JSON, but failed to remove from ChromaDB: {e}")
            else:
                st.warning(f"⚠️ Deleted '{current_tag}' from JSON, but ChromaDB is not available")

with col2:
    # --- Статус синхронизации ---
    st.markdown("### 🔄 Sync Status")
    
    if not db:
        st.error("❌ ChromaDB not available")
    else:
        st.success(f"✅ ChromaDB connected ({len(chroma_ids)} items)")
        
        # Показываем статус синхронизации для каждого тега
        if sync_status:
            synced_count = sum(sync_status.values())
            st.write(f"**Synced: {synced_count}/{len(sync_status)}**")
            
            # Кнопка полной синхронизации
            if st.button("🔄 Sync All Tags"):
                progress_bar = st.progress(0)
                for i, tag in enumerate(tags_data):
                    try:
                        sync_tag_to_chroma(tag, db)
                        progress_bar.progress((i + 1) / len(tags_data))
                        time.sleep(0.1)  # Небольшая задержка
                    except Exception as e:
                        st.error(f"Failed to sync {tag['tag']}: {e}")
                st.success("✅ All tags synced!")
                st.rerun()
            
            # Подробный статус
            if st.checkbox("Show detailed sync status"):
                for tag_name, is_synced in sync_status.items():
                    status_icon = "✅" if is_synced else "❌"
                    st.write(f"{status_icon} {tag_name}")

# --- Сайдбар с категориями и статистикой ---
st.sidebar.markdown("### 📊 Statistics")
st.sidebar.write(f"**Total tags in JSON:** {len(tags_data)}")
if db:
    st.sidebar.write(f"**Records in ChromaDB:** {len(chroma_ids)}")
    st.sidebar.info("💡 Note: ChromaDB may have more records than tags if there are duplicates or old entries")

st.sidebar.markdown("### 📚 Existing Categories")
categories = sorted(set(tag["category"] for tag in tags_data if tag.get("category")))
for cat in categories:
    st.sidebar.write(f"- {cat}")

# --- Дополнительная диагностика ---
if st.sidebar.checkbox("📋 Show ChromaDB Analysis"):
    if db:
        try:
            # Получаем все документы для анализа
            all_data = db.get(include=['documents'])
            
            if all_data and all_data['documents']:
                st.sidebar.markdown("#### 🔍 Document Analysis")
                
                # Анализируем содержимое документов
                tag_docs = []
                other_docs = []
                
                for i, doc in enumerate(all_data['documents']):
                    if doc.startswith("Tag: "):
                        # Извлекаем имя тега из документа
                        lines = doc.split('\n')
                        tag_name = lines[0].replace("Tag: ", "").strip()
                        tag_docs.append(tag_name)
                    else:
                        other_docs.append(doc[:50] + "..." if len(doc) > 50 else doc)
                
                st.sidebar.write(f"**Tag documents:** {len(tag_docs)}")
                if tag_docs:
                    # Показываем найденные теги
                    unique_tags = list(set(tag_docs))
                    st.sidebar.write(f"**Unique tags found:** {len(unique_tags)}")
                    
                    # Проверяем дубликаты
                    duplicates = len(tag_docs) - len(unique_tags)
                    if duplicates > 0:
                        st.sidebar.warning(f"⚠️ Found {duplicates} duplicate entries!")
                    
                    # Показываем первые несколько тегов
                    if st.sidebar.checkbox("Show found tags"):
                        for tag in sorted(unique_tags)[:10]:
                            st.sidebar.write(f"• {tag}")
                        if len(unique_tags) > 10:
                            st.sidebar.write(f"... и ещё {len(unique_tags) - 10}")
                
                if other_docs:
                    st.sidebar.write(f"**Non-tag documents:** {len(other_docs)}")
                    if st.sidebar.checkbox("Show non-tag docs"):
                        for doc in other_docs[:3]:
                            st.sidebar.write(f"• {doc}")
                        
        except Exception as e:
            st.sidebar.error(f"Error analyzing ChromaDB: {e}")

# --- Функции очистки ChromaDB ---
def get_all_chroma_documents():
    """Получает все документы из ChromaDB с их содержимым"""
    try:
        db = get_chroma_db()
        all_data = db.get(include=['documents'])  # Исправлено: убрал 'ids' из include
        return all_data
    except Exception as e:
        st.error(f"Error getting ChromaDB documents: {e}")
        return None

def clean_orphaned_chroma_entries(tags_data):
    """Удаляет записи из ChromaDB, которых нет в JSON"""
    if not db:
        return "ChromaDB not available"
    
    # Получаем все ожидаемые ID из JSON
    expected_ids = {get_tag_id(tag['tag']) for tag in tags_data}
    
    # Получаем все ID из ChromaDB
    all_data = db.get()
    existing_ids = set(all_data['ids'])
    
    # Находим лишние записи
    orphaned_ids = existing_ids - expected_ids
    
    if orphaned_ids:
        try:
            db.delete(ids=list(orphaned_ids))
            return f"Removed {len(orphaned_ids)} orphaned entries from ChromaDB"
        except Exception as e:
            return f"Error removing orphaned entries: {e}"
    else:
        return "No orphaned entries found"

def search_tag_in_chroma(tag_name):
    """Ищет тег в ChromaDB по содержимому документа"""
    try:
        all_data = get_all_chroma_documents()
        if not all_data:
            return []
        
        matches = []
        for i, doc in enumerate(all_data['documents']):
            if tag_name.lower() in doc.lower():
                matches.append({
                    'id': all_data['ids'][i],
                    'document': doc,
                    'exact_match': f"Tag: {tag_name}" in doc
                })
        return matches
    except Exception as e:
        st.error(f"Error searching in ChromaDB: {e}")
        return []

# --- Отладочная информация ---
if st.sidebar.checkbox("🔧 Debug Info"):
    st.sidebar.markdown("### Debug Information")
    st.sidebar.write(f"Current tag: {current_tag}")
    if current_tag:
        tag_id = get_tag_id(current_tag)
        st.sidebar.write(f"Tag ID: {tag_id}")
        if db:
            is_in_chroma = tag_id in chroma_ids
            st.sidebar.write(f"In ChromaDB: {is_in_chroma}")
            
            # Поиск тега в ChromaDB по содержимому
            matches = search_tag_in_chroma(current_tag)
            if matches:
                st.sidebar.write(f"Found {len(matches)} matches in ChromaDB:")
                for match in matches:
                    exact = "🎯" if match['exact_match'] else "📄"
                    st.sidebar.write(f"{exact} ID: {match['id'][:8]}...")

# --- Управление ChromaDB ---
st.sidebar.markdown("### 🛠️ ChromaDB Management")

if st.sidebar.button("🔍 Show All ChromaDB Entries"):
    all_data = get_all_chroma_documents()
    if all_data:
        st.sidebar.write(f"**Total entries in ChromaDB: {len(all_data['ids'])}**")
        
        # Группируем по типу записей
        tag_entries = []
        other_entries = []
        
        for i, doc in enumerate(all_data['documents']):
            entry = {
                'id': all_data['ids'][i],
                'document': doc[:100] + "..." if len(doc) > 100 else doc
            }
            
            if doc.startswith("Tag: "):
                tag_entries.append(entry)
            else:
                other_entries.append(entry)
        
        if tag_entries:
            st.sidebar.write(f"**Tag entries: {len(tag_entries)}**")
            for entry in tag_entries[:5]:  # Показываем только первые 5
                st.sidebar.text(f"ID: {entry['id'][:8]}...")
                st.sidebar.text(f"Doc: {entry['document']}")
                st.sidebar.write("---")
        
        if other_entries:
            st.sidebar.write(f"**Other entries: {len(other_entries)}**")

if st.sidebar.button("🧹 Clean Orphaned Entries"):
    result = clean_orphaned_chroma_entries(tags_data)
    st.sidebar.success(result)
    st.rerun()

if st.sidebar.button("⚠️ Clear All ChromaDB", type="secondary"):
    st.sidebar.warning("⚠️ This will DELETE ALL records from ChromaDB!")
    st.sidebar.write("Use this if you want to start fresh and re-sync all tags")
    if st.sidebar.checkbox("I confirm clearing ALL ChromaDB data"):
        try:
            # Получаем все ID из ChromaDB
            all_data = db.get()
            if all_data['ids']:
                db.delete(ids=all_data['ids'])
                st.sidebar.success(f"✅ Deleted {len(all_data['ids'])} records from ChromaDB")
            else:
                st.sidebar.info("ChromaDB is already empty")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error clearing ChromaDB: {e}")

# --- Поиск конкретного тега ---
st.sidebar.markdown("### 🔎 Search Specific Tag")
search_tag = st.sidebar.text_input("Search tag in ChromaDB:", "")
if search_tag and st.sidebar.button("Search"):
    matches = search_tag_in_chroma(search_tag)
    if matches:
        st.sidebar.write(f"Found {len(matches)} matches:")
        for match in matches:
            exact = "🎯 Exact" if match['exact_match'] else "📄 Partial"
            st.sidebar.write(f"{exact} - ID: {match['id'][:8]}...")
            st.sidebar.text(match['document'][:150] + "...")
            st.sidebar.write("---")
    else:
        st.sidebar.write("No matches found")