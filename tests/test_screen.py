"""Screen plumbing — RepaintBuffer line-splitting + thread safety, BufferConsole
output routing, _NullStatus no-ops."""

import threading

from repl.screen import BufferConsole, RepaintBuffer, ScrollState, _NullStatus


def test_write_splits_complete_lines_and_holds_partial():
    buf = RepaintBuffer()
    buf.write("alpha\nbeta\ngamm")
    assert buf.lines == ["alpha", "beta"]
    assert buf.line_count() == 3  # the partial "gamm" counts as a logical line
    buf.write("a\n")
    assert buf.lines == ["alpha", "beta", "gamma"]
    assert buf.line_count() == 3  # no partial now


def test_view_slices_a_clamped_viewport():
    buf = RepaintBuffer()
    buf.write("\n".join(f"line{i}" for i in range(10)) + "\n")
    assert buf.view(0, 3) == ["line0", "line1", "line2"]
    assert buf.view(8, 5) == ["line8", "line9"]  # clamped at the end
    assert buf.view(-5, 2) == ["line0", "line1"]  # negative top clamps to 0


def test_partial_line_is_visible_in_view():
    buf = RepaintBuffer()
    buf.write("done\nin progress")
    assert buf.view(0, 5) == ["done", "in progress"]


def test_on_change_fires_on_every_write_and_clear():
    hits = []
    buf = RepaintBuffer(on_change=lambda: hits.append(1))
    buf.write("x")
    buf.write("y\n")
    assert len(hits) == 2
    buf.clear()
    assert len(hits) == 3
    assert buf.lines == [] and buf.line_count() == 0


def test_concurrent_writes_do_not_corrupt():
    buf = RepaintBuffer()

    def worker(tag: str):
        for _ in range(200):
            buf.write(f"{tag}\n")

    threads = [threading.Thread(target=worker, args=(t,)) for t in "abcd"]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(buf.lines) == 800
    # every committed line is one of the four whole tags — no interleaving
    assert set(buf.lines) == {"a", "b", "c", "d"}


def test_mark_and_rewind_swap_streamed_content():
    buf = RepaintBuffer()
    buf.write("kept 1\nkept 2\n")
    m = buf.mark()
    assert m == 2
    buf.write("raw token a\nraw token b\n")  # streamed content
    assert buf.line_count() == 4
    buf.rewind_to(m)  # drop the streamed lines
    assert buf.lines == ["kept 1", "kept 2"]
    buf.write("rendered\n")  # re-render in their place
    assert buf.lines == ["kept 1", "kept 2", "rendered"]


def test_null_status_supports_both_call_styles():
    s = _NullStatus()
    s.start()
    s.update("anything")
    s.stop()
    with _NullStatus() as ctx:
        assert ctx is not None


def test_scroll_state_follows_tail_by_default():
    s = ScrollState()
    assert s.follow_tail and s.scroll_top == 0
    # 100 lines in a 20-row pane → tail sits at line 80
    s.on_content(total_lines=100, height=20)
    assert s.scroll_top == 80
    s.on_content(total_lines=140, height=20)
    assert s.scroll_top == 120  # stays pinned to bottom as content grows


def test_scroll_state_page_up_disengages_tail():
    s = ScrollState()
    s.on_content(100, 20)  # at 80, following
    s.page_up(20)
    assert not s.follow_tail
    assert s.scroll_top == 60
    # new content arrives — must NOT yank the viewport back to the bottom
    s.on_content(200, 20)
    assert s.scroll_top == 60


def test_scroll_state_page_down_to_bottom_re_engages_tail():
    s = ScrollState()
    s.on_content(100, 20)
    s.page_up(20)  # at 60, not following
    s.page_down(100, 20)  # back to 80 = bottom
    assert s.scroll_top == 80
    assert s.follow_tail


def test_scroll_state_clamps_and_handles_content_smaller_than_pane():
    s = ScrollState()
    s.page_up(20)  # not following, scroll_top still 0
    s.scroll(delta=50, total_lines=10, height=20)  # everything fits
    assert s.scroll_top == 0
    assert s.follow_tail  # at bottom (which is top) → tailing
    s.to_bottom(10, 20)
    assert s.scroll_top == 0 and s.follow_tail


def test_buffer_console_routes_output_into_the_buffer():
    buf = RepaintBuffer()
    console = BufferConsole(buf, width=80)
    console.print("hello world")
    assert any("hello world" in line for line in buf.lines)
    # status() is neutered, not a real rich spinner
    assert isinstance(console.status("thinking"), _NullStatus)
