"""Agent mode: tool choice, citations across tool calls, triage bypass and the step limit (no network)."""

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage

from physiobot.agent import PhysioAgent, facts_section, is_open
from physiobot.retriever import Passage

PASSAGES = [
    Passage("Tidy's Physiotherapy", 301, "Frozen shoulder (adhesive capsulitis)...", 0.8),
    Passage("Tidy's Physiotherapy", 302, "Pendulum exercises...", 0.7),
]


class ScriptedModel(GenericFakeChatModel):
    """Returns scripted AIMessages in order and records the messages it was shown."""

    seen: list = []

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def call(name, args, id_):
    return {"name": name, "args": args, "id": id_, "type": "tool_call"}


class FakeEmbedder:
    def embed_query(self, q):
        return [0.0] * 768


class FakeRetriever:
    def __init__(self, passages=PASSAGES):
        self.passages = passages
        self.calls = 0

    def search(self, embedding):
        self.calls += 1
        return self.passages


def make(script, retriever=None, max_steps=4):
    model = ScriptedModel(messages=iter(script), seen=[])
    agent = PhysioAgent(FakeEmbedder(), retriever or FakeRetriever(), model=model, max_steps=max_steps)
    return agent, model


def test_is_open_handles_sessions_breaks_and_sunday():
    assert is_open("Saturday", "16:00").startswith("Open")
    assert is_open("monday", "13:30").startswith("Closed at 13:30")
    assert is_open("sunday", "10:00").startswith("Closed: the clinic is closed on Sunday")
    assert is_open("monday", "20:00").startswith("Closed")
    assert "Unknown time" in is_open("monday", "4pm")
    assert "Unknown day" in is_open("someday", "10:00")


def test_facts_section_matches_label_or_lists_topics():
    assert "Home visits" in facts_section("booking")
    assert facts_section("zzz").startswith("No exact match. Available topics:")


def test_agent_uses_hours_tool_without_retrieval():
    retriever = FakeRetriever()
    agent, model = make(
        [
            AIMessage("", tool_calls=[call("clinic_open_at", {"day": "saturday", "time_24h": "16:00"}, "1")]),
            AIMessage("Yes, open on Saturday from 15:00 to 20:00."),
        ],
        retriever=retriever,
    )
    answer, trace = agent.answer("Are you open Saturday at 4 pm?")
    assert trace == ["clinic_open_at"]
    assert retriever.calls == 0
    assert answer.sources == []
    tool_msg = [m for m in model.seen[1] if isinstance(m, ToolMessage)][0]
    assert tool_msg.content.startswith("Open")


def test_agent_combines_textbook_and_clinic_facts_with_citations():
    agent, model = make(
        [
            AIMessage(
                "",
                tool_calls=[
                    call("search_textbook", {"query": "frozen shoulder exercises"}, "1"),
                    call("clinic_facts", {"topic": "treatments"}, "2"),
                ],
            ),
            AIMessage("Pendulum exercises can help [2]. The clinic treats shoulder pain."),
        ]
    )
    answer, trace = agent.answer("What helps a frozen shoulder, and do you treat it?")
    assert trace == ["search_textbook", "clinic_facts"]
    assert answer.sources == [{"source": "Tidy's Physiotherapy", "page": 302}]


def test_passage_numbers_continue_across_searches():
    agent, model = make(
        [
            AIMessage("", tool_calls=[call("search_textbook", {"query": "a"}, "1")]),
            AIMessage("", tool_calls=[call("search_textbook", {"query": "b"}, "2")]),
            AIMessage("See [3]."),
        ]
    )
    answer, _ = agent.answer("two searches")
    second = [m for m in model.seen[2] if isinstance(m, ToolMessage)][-1].content
    assert second.startswith("[3]") and "[4]" in second
    assert answer.sources == [{"source": "Tidy's Physiotherapy", "page": 301}]


def test_red_flag_skips_the_model():
    agent, model = make([AIMessage("should not be used")])
    answer, trace = agent.answer("I have chest pain spreading to my left arm")
    assert answer.urgent and trace == [] and model.seen == []


def test_unknown_tool_and_step_limit():
    loop = [AIMessage("", tool_calls=[call("delete_everything", {}, str(i))]) for i in range(3)]
    agent, model = make(loop, max_steps=3)
    answer, trace = agent.answer("hello")
    assert trace == ["delete_everything"] * 3
    assert "too many steps" in answer.answer
    assert "Unknown tool" in [m for m in model.seen[1] if isinstance(m, ToolMessage)][0].content
