import json, os, time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from dotenv import dotenv_values
from retrieval import ConfigurationError, ProviderError


def streamlit_secret(name):
    """Read a Streamlit-managed secret without requiring one for local .env use."""
    try:
        import streamlit as st
        return st.secrets.get(name)
    except Exception:
        return None


class LLMClient:
    def __init__(self):
        path = Path(os.getenv('MOVIE_ENV_FILE', Path(__file__).with_name('.env')))
        values = dotenv_values(path) if path.is_file() else {}
        # Environment variables take precedence, then local .env, then Streamlit Cloud secrets.
        self.key = os.getenv('OPENAI_API_KEY') or values.get('OPENAI_API_KEY') or streamlit_secret('OPENAI_API_KEY')
        self.model = os.getenv('OPENAI_MODEL') or values.get('OPENAI_MODEL') or streamlit_secret('OPENAI_MODEL') or 'gpt-5.4-mini'
        if not self.key:
            raise ConfigurationError('Configure OPENAI_API_KEY in .env or Streamlit app secrets.')
        self.metrics = []

    def create(self, **kwargs):
        body = {'model':self.model, 'store':False, 'max_output_tokens':4000, **kwargs}
        if self.model.startswith('gpt-5'):
            body['reasoning'] = {'effort':'low'}
        req = Request('https://api.openai.com/v1/responses',data=json.dumps(body).encode(),
                      headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json'})
        start = time.monotonic()
        try:
            with urlopen(req, timeout=90) as response:
                data = json.load(response)
        except HTTPError as exc:
            code = ''
            try:
                raw = json.load(exc).get('error',{}).get('code','')
                if isinstance(raw,str) and raw.replace('_','').isalnum(): code=raw[:80]
            except (ValueError,AttributeError,TypeError): pass
            raise ProviderError(f'OpenAI HTTP {exc.code} ({code or "request_failed"}); check key, API credits, or model access.') from None
        except (URLError, OSError, ValueError):
            raise ProviderError('OpenAI connection failed or returned invalid JSON; check the network.') from None
        self.metrics.append({'model':data.get('model',self.model),'response_id':data.get('id',''),
                             'seconds':round(time.monotonic()-start,3),'usage':data.get('usage',{})})
        if data.get('status') != 'completed':
            raise ProviderError('The model response did not complete; partial output was not used.')
        return data


def output_text(data):
    text='\n'.join(p['text'] for x in data.get('output',[]) if x.get('type')=='message'
                   for p in x.get('content',[]) if p.get('type')=='output_text')
    if not text: raise ProviderError('The model returned no usable text (it may have refused the request).')
    return text
