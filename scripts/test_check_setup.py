import check_setup


def test_fast_checks_pass():
    for c in check_setup.FAST:
        assert c() == [], c.__name__
