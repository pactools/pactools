import time

from pactools.utils.testing import assert_true, assert_false
from pactools.utils.progress_bar import ProgressBar


def test_progress_bar():
    n = 10
    app = ProgressBar(title='Testing progressBar', max_value=n - 1,
                      spinner=True)
    assert_false(app.closed)
    for k in range(n):
        app.update_with_increment_value(1)
        time.sleep(0.001)
    assert_true(app.closed)

    n = 10
    app = ProgressBar(title='Testing progressBar', max_value=n - 1,
                      spinner=True)
    app.update(n)
    assert_true(app.closed)
