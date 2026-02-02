import streamlit as st
import tempfile
import whisper
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Voice Bot 100x – Mahesh Kumar Jangid",
    page_icon="🎙",
    layout="centered"
)

st.title("🎙 Voice Bot 100x")
st.caption("Mahesh Kumar Jangid — Generative AI Engineer Persona")

# ---------------- MEMORY INIT ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

whisper_model = load_whisper()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.5,
    groq_api_key=st.secrets["GROQ_API_KEY"]
)

SYSTEM_PROMPT = """
You are Mahesh Kumar Jangid, a Generative AI Engineer candidate.
Answer in first person.
Be confident, concise, and professional.
"""

# ---------------- INPUT UI ----------------
audio = st.audio_input("🎤 Speak (optional)")
text = st.text_input("✍️ Or type your question")

# ---------------- SUBMIT ----------------
if st.button("Submit"):
    user_text = None

    # 🎤 Voice → Text
    if audio:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio.read())
            tmp_path = tmp.name

        result = whisper_model.transcribe(tmp_path)
        user_text = result["text"].strip()

    # ✍️ Text fallback
    if text:
        user_text = text

    if not user_text:
        st.error("Please speak or type something.")
    else:
        # Save user message to memory
        st.session_state.chat_history.append(
            HumanMessage(content=user_text)
        )

        # Build conversation with memory
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        messages.extend(st.session_state.chat_history)

        with st.spinner("Thinking..."):
            response = llm.invoke(messages)

        # Save AI response to memory
        st.session_state.chat_history.append(
            AIMessage(content=response.content)
        )

# ---------------- DISPLAY CHAT ----------------
st.markdown("---")
st.markdown("### 🧠 Conversation Memory")

for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.markdown(f"** You:** {msg.content}")
    else:
        st.markdown(f"** AI:** {msg.content}")

# ---------------- RESET ----------------
if st.button("🧹 Reset Memory"):
    st.session_state.chat_history = []
    st.experimental_rerun()
