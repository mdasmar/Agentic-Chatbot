from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from chatbot import workflow,get_threads,rag_docs
from langgraph.types import Command
import streamlit as st
import uuid
import tempfile
import os

def gen_thread_id():
    return str(uuid.uuid4())

def add_thread(thread_id):
    if thread_id not in st.session_state["chat_thread"]:
        st.session_state["chat_thread"].append(thread_id)

def reset_chat():
    st.session_state['thread_id']=gen_thread_id()
    st.session_state['msg_history']=[]
    add_thread(st.session_state['thread_id'])

def load_conversation(thread_id):
    state=workflow.get_state(config={"configurable":{"thread_id":thread_id}})
    return state.values.get("messages",[])

def get_pending_interrupt(thread_id):
    """
    Return the first unresolved LangGraph interrupt for a thread.
    Returns:
        The pending Interrupt object, or None.
    """
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state_snapshot = workflow.get_state(config)
        direct_interrupts = getattr(state_snapshot,"interrupts",()) or ()
        if direct_interrupts:
            return direct_interrupts[0]

        tasks = getattr(state_snapshot,"tasks",()) or ()
        for task in tasks:
            task_interrupts = getattr(task,"interrupts",()) or ()
            if task_interrupts:
                return task_interrupts[0]

    except Exception:
        return None
    return None


def save_pending_interrupt(thread_id, interrupt_object):
    """
    Save the pending interrupt information inside Streamlit state.
    """
    st.session_state["pending_hitl"] = {"thread_id": thread_id,"prompt": str(interrupt_object.value)}

def sync_pending_interrupt(thread_id):
    """
    Synchronize Streamlit HITL state with the LangGraph checkpoint.

    This allows a pending approval request to reappear after:
    - a Streamlit rerun
    - a browser refresh
    - switching between conversations
    """
    pending_interrupt = get_pending_interrupt(thread_id)
    if pending_interrupt is not None:
        save_pending_interrupt(thread_id,pending_interrupt)
    else:
        current_pending = st.session_state.get("pending_hitl")
        if (current_pending is not None and current_pending.get("thread_id") == thread_id):
            st.session_state["pending_hitl"] = None

def resume_hitl_execution(decision):
    """
    Resume an interrupted LangGraph execution.
    Args:
        decision:
            "yes" approves the stock purchase.
            "no" rejects the stock purchase.
    """
    pending_hitl = st.session_state.get("pending_hitl")
    if not pending_hitl:
        st.warning("There is no pending action to approve or reject.")
        return

    interrupted_thread_id = pending_hitl["thread_id"]
    resume_config={"configurable": {"thread_id": interrupted_thread_id},
        "metadata":{"thread_id": interrupted_thread_id},
        "run_name": "hitl_resume_trace",}
    try:
        with st.chat_message("bot"):
            status_holder = {"box": st.status("🔄 Resuming the requested action...",expanded=True)}
            def resumed_ai_only_stream():
                for message_chunk, metadata in workflow.stream(
                    Command(resume=decision),
                    config=resume_config,
                    stream_mode="messages",
                ):
                    if isinstance(message_chunk,ToolMessage):
                        tool_name = getattr(message_chunk,"name","tool")
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )
                    if isinstance(message_chunk,AIMessage):
                        if message_chunk.content:
                            yield message_chunk.content

            resumed_ai_message = st.write_stream(resumed_ai_only_stream())
            next_interrupt = get_pending_interrupt(
                interrupted_thread_id
            )
            if next_interrupt is not None:
                save_pending_interrupt(
                    interrupted_thread_id,
                    next_interrupt
                )
                status_holder["box"].update(
                    label="⚠️ Another approval is required",
                    state="complete",
                    expanded=False
                )
            else:
                st.session_state["pending_hitl"] = None
                status_holder["box"].update(
                    label="✅ Action completed",
                    state="complete",
                    expanded=False
                )
        if resumed_ai_message:
            st.session_state["msg_history"].append({
                "role": "bot",
                "content": resumed_ai_message
            })
        st.rerun()
    except Exception as error:
        st.error(f"Could not resume the requested action: {error}")

st.set_page_config(
    page_title="Agentic Chatbot"
)

st.title("Agentic Chatbot")


if 'msg_history' not in st.session_state:
    st.session_state['msg_history']=[]

for msg in st.session_state['msg_history']:
    with st.chat_message(msg['role']):
        st.text(msg['content'])

if "pending_hitl" not in st.session_state:
    st.session_state["pending_hitl"] = None

if 'chat_thread' not in st.session_state:
    st.session_state['chat_thread']=get_threads()

if 'thread_id' not in st.session_state:
    st.session_state['thread_id']=gen_thread_id()

add_thread(st.session_state['thread_id'])

sync_pending_interrupt(st.session_state["thread_id"])

st.sidebar.title("Chats")

if st.sidebar.button("New chat"):
    reset_chat()
    st.rerun()


for thread_id in st.session_state['chat_thread'][::-1]:
    if st.sidebar.button(str(thread_id),key=thread_id):
        st.session_state['thread_id']=thread_id

        messages=load_conversation(thread_id)
        temp_msg=[]
        for msg in messages:
            if isinstance(msg,HumanMessage):
                role='user'

            elif isinstance(msg,AIMessage):
                role='bot'
            else:
                continue

            temp_msg.append({"role":role,"content":msg.content})

        st.session_state['msg_history']=temp_msg
        sync_pending_interrupt(thread_id)
        st.rerun()

pending_hitl = st.session_state.get("pending_hitl")

current_thread_has_pending_hitl = (
    pending_hitl is not None
    and pending_hitl.get("thread_id")
    == st.session_state["thread_id"]
)

if current_thread_has_pending_hitl:
    st.warning(
        "🧑 Human approval required\n\n"
        f"{pending_hitl['prompt']}"
    )

    approve_column, reject_column = st.columns(2)
    with approve_column:

        if st.button(
            "✅ Approve Purchase",
            key=f"approve_{st.session_state['thread_id']}",
            type="primary",
            use_container_width=True
        ):
            resume_hitl_execution("yes")
    with reject_column:
        if st.button(
            "❌ Reject Purchase",
            key=f"reject_{st.session_state['thread_id']}",
            use_container_width=True):
            resume_hitl_execution("no")

submission = st.chat_input(
    "Ask anything!",
    accept_file=True,
    file_type=["pdf"],
    disabled=current_thread_has_pending_hitl
)

user_input = None

if submission:
    user_input = submission.text
    uploaded_files = submission.files
    if uploaded_files:
        uploaded_pdf=uploaded_files[0]
        temporary_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as temporary_file:
                temporary_file.write(uploaded_pdf.getvalue())
                temporary_file_path = temporary_file.name

            with st.spinner(f"Processing {uploaded_pdf.name}..."):
                rag_docs(temporary_file_path)

            st.toast(f"{uploaded_pdf.name} processed successfully.",icon="✅")
        except Exception as error:
            st.error(f"PDF processing failed: {error}")
        finally:
            if (temporary_file_path and os.path.exists(temporary_file_path)):
                os.remove(temporary_file_path)

if user_input:
    st.session_state['msg_history'].append({'role':'user','content':user_input})
    with st.chat_message("user"):
        st.text(user_input)

    config={'configurable':{'thread_id':st.session_state['thread_id']},
            'metadata':{'thread_id':st.session_state['thread_id']},
            'run_name':'chat_trace'}

    with st.chat_message("bot"):
        status_holder = {"box": None}

        def chat_stream():
            for message_chunk, metadata in workflow.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages"
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")

                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(f"🔧 Using `{tool_name}` …",
                            expanded=True
                        )
                    else:
                        status_holder["box"].update(label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

            pending_interrupt = get_pending_interrupt(st.session_state["thread_id"])
            if pending_interrupt is not None:
                save_pending_interrupt(
                    st.session_state["thread_id"],
                    pending_interrupt
                )

                yield (
                    "\n\n⚠️ This stock purchase requires your approval. "
                    "Use the Approve Purchase or Reject Purchase "
                    "button below."
                )

        response = st.write_stream(chat_stream())

        if status_holder["box"] is not None:
            if get_pending_interrupt(st.session_state["thread_id"]) is not None:

                status_holder["box"].update(
                    label="⏸️ Waiting for human approval",
                    state="complete",
                    expanded=False
                )
            else:
                status_holder["box"].update(
                    label="✅ Tool finished",
                    state="complete",
                    expanded=False
                )            

    st.session_state['msg_history'].append({'role':'bot','content':response})

    if (st.session_state.get("pending_hitl") is not None
        and st.session_state["pending_hitl"].get("thread_id")
        == st.session_state["thread_id"]
    ):
        st.rerun()




