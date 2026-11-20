from ullm.tokenizer import (
    ASSISTANT,
    END,
    ENDOFTEXT,
    USER,
    Message,
    StreamDecoder,
    decode,
    encode,
    render_chat,
)


def test_special_token_ids():
    assert (ENDOFTEXT, USER, ASSISTANT, END) == (50256, 50257, 50258, 50259)


def test_round_trip():
    text = "The quick brown fox, jumps over 3 lazy dogs! Ünïcödé 🦊"
    assert decode(encode(text)) == text


def test_matches_gpt2_encoding():
    assert encode("Hello world") == [15496, 995]


def test_special_token_text_in_input_is_plain_text():
    for text in ["<|end|>", "<|user|>", "<|assistant|>", "<|endoftext|>"]:
        ids = encode(text)
        assert not {ENDOFTEXT, USER, ASSISTANT, END} & set(ids)
        assert decode(ids) == text


def test_decode_shows_special_tokens():
    assert decode([USER, END]) == "<|user|><|end|>"


def test_render_chat_layout_and_mask():
    messages = [
        Message("user", "Hi"),
        Message("assistant", "Hello there"),
        Message("user", "Bye"),
        Message("assistant", "Goodbye"),
    ]
    ids, mask = render_chat(messages)

    hi, hello, bye, goodbye = (encode(m.content) for m in messages)
    assert ids == (
        [USER, *hi, END, ASSISTANT, *hello, END] + [USER, *bye, END, ASSISTANT, *goodbye, END]
    )

    def user_turn(content: list[int]) -> list[int]:
        return [0] * (len(content) + 2)

    def assistant_turn(content: list[int]) -> list[int]:
        return [0] + [1] * (len(content) + 1)

    assert mask == user_turn(hi) + assistant_turn(hello) + user_turn(bye) + assistant_turn(goodbye)


def test_render_chat_of_open_user_turn_has_no_loss():
    ids, mask = render_chat([Message("user", "Hi")])
    assert ids == [USER, *encode("Hi"), END]
    assert mask == [0] * len(ids)


def test_stream_decoder_never_splits_characters():
    text = "naïve café 日本語 🦊🦊 done"
    ids = encode(text)
    decoder = StreamDecoder()
    pieces = [decoder.push(token) for token in ids] + [decoder.flush()]
    assert "".join(pieces) == text
    assert all("�" not in piece for piece in pieces)
