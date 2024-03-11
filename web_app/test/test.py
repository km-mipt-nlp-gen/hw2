import sys
import os
import pytest


@pytest.fixture(scope="session")
def app():
    run_web_app = import_web_app_module()
    _app = run_web_app(True)
    return _app


def import_web_app_module():
    web_app_src_path = os.getenv('WEB_APP_SRC_PATH')
    assert web_app_src_path is not None, 'Путь web_app_src_path должен быть установлен.'
    sys.path.append(web_app_src_path)
    from run_web_app_script import run_web_app
    return run_web_app


@pytest.fixture(scope="session")
def client(app):
    with app.test_client() as client:
        yield client


@pytest.fixture(autouse=True)
def clear_chat_history_before_and_after_tests(client):
    client.delete("/clear")
    yield
    client.delete("/clear")


@pytest.mark.parametrize("endpoint,expected_length", [
    ("/top_cos_sim_bi_cr", 6),
    ("/top_l2_bi_cr", 6),
    ("/top_l2_psa_bi_cr", 6),
    ("/top_cr", 6)
])
def test_that_expected_length_when_find_top_3_times(client, endpoint, expected_length):
    response = None
    for _ in range(3):
        response = client.post(endpoint, json={"query": "test", "user": "test_user"})

    assert response.status_code == 200
    assert len(response.json['response']) == expected_length


@pytest.mark.parametrize("endpoint,expected_length", [
    ("/top_cos_sim_bi_cr", 2),
    ("/top_l2_bi_cr", 2),
    ("/top_l2_psa_bi_cr", 2),
    ("/top_cr", 2)
])
def test_that_expected_length_when_find_top(client, endpoint, expected_length):
    response = client.post(endpoint, json={"query": "test", "user": "test_user"})

    assert response.status_code == 200
    assert len(response.json['response']) == expected_length


@pytest.mark.parametrize("endpoint", [
    ("/top_cos_sim_bi_cr"),
    ("/top_l2_bi_cr"),
    ("/top_l2_psa_bi_cr"),
    ("/top_cr")
])
def test_that_clear_chat_when_find_top_then_clear(client, endpoint):
    client.post(endpoint, json={"query": "test", "user": "test_user"})
    client.delete('/clear')
    response = client.get('/chat')

    assert response.status_code == 200
    assert len(response.json['response']) == 0

# TODO можно добавить новую функциональность в будущем
# @pytest.mark.parametrize("top_n,endpoint", [
#     (1, "/top_cos_sim_bi_cr"), (5, "/top_cos_sim_bi_cr"), (10, "/top_cos_sim_bi_cr"),
#     (1, "/top_l2_bi_cr"), (5, "/top_l2_bi_cr"), (10, "/top_l2_bi_cr"),
#     (1, "/top_l2_psa_bi_cr"), (5, "/top_l2_psa_bi_cr"), (10, "/top_l2_psa_bi_cr"),
#     (1, "/top_cr"), (5, "/top_cr"), (10, "/top_cr"),
# ])
# def test_top_n_responses(client, top_n, endpoint):
#     response = client.post(endpoint, json={"query": "test", "user": "test_user", "top_n": top_n})
#     assert response.status_code == 200
