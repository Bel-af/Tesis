import dotenv
import os
import json


def _normalize_prefix(prefix_value):
    if not prefix_value:
        return ''
    return prefix_value if prefix_value.startswith('/') else f'/{prefix_value}'


def _build_base_url(scheme_value, domain_value, port_value, prefix_value):
    normalized_prefix = _normalize_prefix(prefix_value)
    normalized_scheme = (scheme_value or 'http').strip()
    normalized_domain = (domain_value or 'localhost').strip()
    normalized_port = str(port_value or '').strip()

    if normalized_port and normalized_port not in ('80', '443'):
        host = f"{normalized_domain}:{normalized_port}"
    else:
        host = normalized_domain

    return f"{normalized_scheme}://{host}{normalized_prefix}"


class ENVIRONMENT:
    def __init__(self):
        project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
        dotenv_path = os.path.join(project_dir, '.env')
        dotenv.load_dotenv(dotenv_path, override=False)
        self.scheme = os.getenv("SCHEME", "http")
        self.domain = os.getenv("DOMAIN", "localhost")
        self.port = os.getenv("PORT", "5000")
        self.prefix = _normalize_prefix(os.getenv("PREFIX", ""))
        self.public_base_url = os.getenv("PUBLIC_BASE_URL", "").rstrip('/')

    def get_instance(self):
        if not hasattr(self, "_instance"):
            self._instance = ENVIRONMENT()
        return self._instance

    def getDomain(self):
        return self.domain

    def getPort(self):
        return self.port

    def getPrefix(self):
        return self.prefix

    def getScheme(self):
        return self.scheme

    def getPublicBaseUrl(self):
        return self.public_base_url


_environment = ENVIRONMENT().get_instance()
scheme = _environment.getScheme()
domain = _environment.getDomain()
port = _environment.getPort()
prefix = _environment.getPrefix()
public_base_url = _environment.getPublicBaseUrl()


def get_runtime_base_url():
    if public_base_url:
        return public_base_url
    return _build_base_url(scheme, domain, port, prefix)


swagger_config_endpoint = f"{get_runtime_base_url()}/swagger-config"


def build_swagger_config_json():
    config_file_path = 'static/swagger/config.json'

    with open(config_file_path, 'r') as file:
        config_data = json.load(file)

    local_server_url = _build_base_url('http', 'localhost', port, prefix)
    runtime_server_url = get_runtime_base_url()

    servers = [{"url": local_server_url}]
    if runtime_server_url != local_server_url:
        servers.append({"url": runtime_server_url})

    config_data['servers'] = servers

    new_config_file_path = 'static/swagger/config.json'

    with open(new_config_file_path, 'w') as new_file:
        json.dump(config_data, new_file, indent=2)
