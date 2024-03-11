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
    ("/gpt2", 6)
])
def test_that_expected_length_when_find_top_3_times(client, endpoint, expected_length):
    response = None
    for _ in range(3):
        response = client.post(endpoint, json={"query": "test", "user": "test_user"})

    assert response.status_code == 200
    assert len(response.json['response']) == expected_length


@pytest.mark.parametrize("endpoint,expected_length", [
    ("/gpt2", 2)
])
def test_that_expected_length_when_find_top(client, endpoint, expected_length):
    response = client.post(endpoint, json={"query": "test", "user": "test_user"})

    assert response.status_code == 200
    assert len(response.json['response']) == expected_length


@pytest.mark.parametrize("endpoint", [
    ("/gpt2")
])
def test_that_clear_chat_when_find_top_then_clear(client, endpoint):
    client.post(endpoint, json={"query": "test", "user": "test_user"})
    client.delete('/clear')
    response = client.get('/chat')

    assert response.status_code == 200
    assert len(response.json['response']) == 0