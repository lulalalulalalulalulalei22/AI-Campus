import streamlit as st
import os
import tempfile
import base64
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI

# --- 1. 页面配置 ---
st.set_page_config(
    page_title="Morandi AI Space",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded" # 保持默认展开，但现在可以方便地收放了
)

# 魔法加速
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# --- 2. 核心函数 (辅助工具) ---

def image_to_base64(uploaded_file):
    """图片转Base64工具"""
    if uploaded_file is None: return None
    try:
        # 限制一下图片大小，防止Base64过长导致卡顿
        return f"data:image/png;base64,{base64.b64encode(uploaded_file.getvalue()).decode('utf-8')}"
    except Exception as e:
        st.error(f"图片上传失败: {e}")
        return None

def optimize_prompt(user_input, api_key):
    """✨ 提示词优化魔法"""
    if not api_key or not user_input: return user_input
    try:
        client = ChatOpenAI(
            model_name="deepseek-chat",
            openai_api_key=api_key,
            openai_api_base="https://api.deepseek.com",
            temperature=0.7
        )
        meta_prompt = f"""
        Role: Senior Prompt Engineer.
        Task: Rewrite the user's simple description into a detailed, professional System Prompt.
        Style: Clear, structured, professional, but maintain a helpful tone.
        Output: Directly output the rewritten prompt content.
        
        User's description: "{user_input}"
        """
        with st.spinner("🌿 正在精心编织你的设定..."):
            response = client.invoke(meta_prompt)
            return response.content
    except Exception as e:
        st.error(f"优化服务暂时不可用: {e}")
        return user_input

# --- 3. CSS 魔法：莫兰迪 + Ins风 ---
# Ins风背景图 (扁平、植物、线条感)
INS_BG_URL = "https://images.unsplash.com/photo-1623697899812-4b367878063e?q=80&w=2670&auto=format&fit=crop"

st.markdown(f"""
<style>
    /* === 全局莫兰迪色系定义 === */
    :root {{
        --morandi-bg: #F7F4EF;       /* 奶油米白背景 */
        --morandi-sidebar: #EAEFE9;  /* 灰豆绿侧边栏 */
        --morandi-user: #EED8D8;     /* 脏粉色气泡 */
        --morandi-ai: #FBF9F4;       /* 燕麦色气泡 */
        --morandi-text: #6B5E52;     /* 深棕灰色文字 */
        --morandi-accent: #AAB7B0;   /* 莫兰迪绿装饰 */
    }}

    /* 应用整体 */
    .stApp {{
        background-image: url("{INS_BG_URL}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        /* 使用更现代干净的字体 */
        font-family: 'Helvetica Neue', Helvetica, 'PingFang SC', 'Microsoft YaHei', sans-serif;
        color: var(--morandi-text);
    }}

    /* 隐藏多余元素，但保留 Header 以便操作侧边栏 */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{
        background: transparent !important; /* 让顶部导航栏透明 */
    }}
    
    /* 侧边栏美化 - 莫兰迪绿 */
    [data-testid="stSidebar"] {{
        background-color: var(--morandi-sidebar) !important;
        border-right: 1px solid #D8E0DC;
    }}
    
    /* 动画定义 (保持之前的呼吸感) */
    @keyframes popIn {{
        0% {{ opacity: 0; transform: translateY(10px); }}
        100% {{ opacity: 1; transform: translateY(0); }}
    }}
    
    /* 聊天气泡 - Ins风简约圆角 */
    .stChatMessage {{
        padding: 15px !important;
        border-radius: 18px !important;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.03);
        animation: popIn 0.4s ease-out forwards;
        border: none !important;
    }}

    /* 用户气泡 (脏粉色) */
    .stChatMessage[data-testid="user-message"] {{
        background-color: var(--morandi-user);
        margin-left: 20%;
        border-bottom-right-radius: 4px !important; /* 一个小角标 */
    }}
    
    /* AI 气泡 (燕麦色) */
    .stChatMessage[data-testid="assistant-message"] {{
        background-color: var(--morandi-ai);
        margin-right: 20%;
        border-bottom-left-radius: 4px !important;
    }}
    
    /* 输入框和按钮美化 */
    .stTextInput input, .stTextArea textarea {{
        background-color: #FFFFFF !important;
        border: 1px solid #E0E0E0 !important;
        border-radius: 12px !important;
        color: var(--morandi-text) !important;
    }}
    .stButton button {{
        background-color: var(--morandi-accent) !important;
        color: white !important;
        border-radius: 10px !important;
        border: none !important;
        transition: all 0.3s;
    }}
    .stButton button:hover {{
        opacity: 0.8;
        transform: translateY(-2px);
    }}

    /* 右侧卡片样式 */
    .persona-card {{
        background: rgba(255,255,255,0.8);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
        border: 1px solid rgba(255,255,255,0.5);
    }}

</style>
""", unsafe_allow_html=True)

# --- 4. 侧边栏逻辑 ---
with st.sidebar:
    st.header("🌿 Space Control")
    
    # 新手引导折叠面板
    with st.expander("📖 使用小贴士", expanded=False):
        st.markdown("""
        <div style="font-size: 0.9em; color: #888;">
        1. 填入 DeepSeek API Key。<br>
        2. 选择或创建一个 AI 伙伴。<br>
        3. 上传 PDF 知识库（可选）。<br>
        4. 开始享受对话。
        </div>
        """, unsafe_allow_html=True)

    # === 智能 Key 管理 (修改版) ===
    # 1. 尝试从 secrets 获取 Key (容错处理：如果没有文件或 Key 为空，返回空字符串)
    try:
        file_key = st.secrets.get("deepseek_api_key", "")
    except FileNotFoundError:
        file_key = ""

    # 2. 逻辑判断：如果文件里没填 Key，就在界面显示输入框
    if not file_key:
        api_key = st.text_input("🔑 API Key", type="password", help="检测到配置文件为空，请手动输入密钥")
    else:
        api_key = file_key
        st.success("✅ 已从配置文件自动加载 Key")
    st.divider()

    # === 用户自身形象设置 ===
    st.subheader("🧑‍🎨 你的模样")
    user_avatar_file = st.file_uploader("上传你的头像 (可选)", type=["png", "jpg"], key="user_av")
    if user_avatar_file:
        st.session_state.user_avatar_base64 = image_to_base64(user_avatar_file)
    # 默认用户头像 (Ins风线条小人)
    user_final_avatar = st.session_state.get("user_avatar_base64", "https://cdn-icons-png.flaticon.com/512/10453/10453453.png")

    st.divider()

    # === AI 伙伴设置 (核心功能修复) ===
    mode = st.radio("🌱 选择伙伴模式", ["📚 Ins风预设", "✨ 自定义创作"])
    
    # 状态初始化
    if "custom_prompt" not in st.session_state: st.session_state.custom_prompt = ""
    if "custom_ai_img" not in st.session_state: st.session_state.custom_ai_img = None

    # Ins风预设角色 (更换了更现代的插图)
    PRESETS = {
        "Math Tutor (植物系)": {"prompt": "You are a gentle math tutor...", "image": "https://cdn-icons-png.flaticon.com/512/10453/10453377.png"},
        "Code Geek (极简风)": {"prompt": "You are a senior coding architect...", "image": "https://cdn-icons-png.flaticon.com/512/10453/10453428.png"}
    }

    if mode == "📚 Ins风预设":
        role = st.selectbox("选择预设", list(PRESETS.keys()))
        prompt_text = PRESETS[role]["prompt"]
        ai_current_img = PRESETS[role]["image"]
        name = role.split(" ")[0]
    else:
        # === 自定义模式 (功能补全!) ===
        st.write("🛠️ **创作工坊**")
        name = st.text_input("伙伴昵称", value="My AI", placeholder="例如：学习搭子")
        
        # 1. 提示词优化区
        col1, col2 = st.columns([4, 1])
        with col1:
            raw_prompt = st.text_area("人设简单描述", height=70, placeholder="例如：一个很会鼓励人的雅思陪练", key="raw_p")
        with col2:
            st.write("") # 占位
            st.write("")
            if st.button("✨", help="点击优化提示词"):
                st.session_state.custom_prompt = optimize_prompt(raw_prompt, api_key)
                st.rerun()
        
        prompt_text = st.text_area("最终设定指令", value=st.session_state.custom_prompt, height=100, key="final_p")

        # 2. 【重要修复】AI形象上传功能
        st.markdown("🖼️ **AI形象设定**")
        custom_ai_file = st.file_uploader("上传AI的立绘/照片", type=["png", "jpg"], key="custom_ai_file")
        if custom_ai_file:
             st.session_state.custom_ai_img = image_to_base64(custom_ai_file)
        
        # 决定最终的 AI 图片：如果有上传就用上传的，否则用默认 Ins 风图标
        ai_current_img = st.session_state.get("custom_ai_img") or "https://cdn-icons-png.flaticon.com/512/10453/10453438.png"

    st.divider()
    pdf = st.file_uploader("📘 添加知识库 (PDF)", type="pdf")
    if st.button("🛁 清空当前对话"):
        st.session_state.messages = []
        st.rerun()

# --- 5. 后端逻辑 (RAG) ---
if pdf and api_key:
    fid = f"{pdf.name}"
    if "fid" not in st.session_state or st.session_state.fid != fid:
        with st.spinner("🌿 正在吸收知识养分..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
                    tf.write(pdf.getvalue())
                loader = PyPDFLoader(tf.name)
                splits = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(loader.load())
                vs = Chroma.from_documents(documents=splits, embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
                st.session_state.vs = vs
                st.session_state.fid = fid
                st.toast(f"✅ {name} 已准备好!", icon="🍃")
            except Exception as e: st.error(str(e))

# --- 6. 页面主体布局 ---
c1, c2 = st.columns([7, 3])

with c2:
    # 右侧卡片 (莫兰迪风格)
    st.markdown(f"""
    <div class="persona-card">
        <img src="{ai_current_img}" style="width: 120px; height: 120px; object-fit: cover; border-radius: 40%; border: 3px solid #FFF; box-shadow: 0 8px 20px rgba(167, 183, 176, 0.2);">
        <h3 style="color: #6B5E52; margin: 15px 0 5px;">{name}</h3>
        <div style="font-size: 0.85em; color: #8C9690; background: #F0F4F1; padding: 6px 12px; border-radius: 15px; display: inline-block;">
             Status: {"🌿 Active" if pdf else "⚪ Idle"}
        </div>
    </div>
    """, unsafe_allow_html=True)

with c1:
    st.subheader(f"💬 Chat with {name}")
    if "messages" not in st.session_state: st.session_state.messages = []
    
    for msg in st.session_state.messages:
        # 使用正确的头像
        av = ai_current_img if msg["role"] == "assistant" else user_final_avatar
        st.chat_message(msg["role"], avatar=av).write(msg["content"])

    if q := st.chat_input("Type something softly..."):
        if not api_key: st.stop()
        st.session_state.messages.append({"role": "user", "content": q})
        st.chat_message("user", avatar=user_final_avatar).write(q)
        
        with st.chat_message("assistant", avatar=ai_current_img):
            with st.spinner("..."):
                ctx = ""
                if "vs" in st.session_state and pdf:
                    try:
                        res = st.session_state.vs.similarity_search(q, k=3)
                        ctx = "\n".join([d.page_content for d in res])
                    except: pass
                
                full_input = f"【System】\n{prompt_text}\n\n【Knowledge】\n{ctx}\n\n【User】\n{q}"
                try:
                    llm = ChatOpenAI(model_name="deepseek-chat", openai_api_key=api_key, openai_api_base="https://api.deepseek.com", temperature=0.7)
                    ans = llm.invoke(full_input)
                    st.write(ans.content)
                    st.session_state.messages.append({"role": "assistant", "content": ans.content})
                except Exception as e: st.error(e)