import io,json,os,unittest
from unittest.mock import patch
from urllib.error import HTTPError
from llm_client import LLMClient,ProviderError
class TransportTests(unittest.TestCase):
 def setUp(self):
  p=patch.dict(os.environ,{'OPENAI_API_KEY':'test-secret-not-a-real-key','OPENAI_MODEL':'gpt-5-mini','MOVIE_ENV_FILE':'/nonexistent'});p.start();self.addCleanup(p.stop)
 def test_request_payload_and_usage(self):
  body={'status':'completed','output':[],'id':'r_test','model':'gpt-5-mini','usage':{'input_tokens':10}}
  with patch('llm_client.urlopen',return_value=io.BytesIO(json.dumps(body).encode())) as call:
   c=LLMClient();c.create(input='test')
   req=call.call_args[0][0];sent=json.loads(req.data)
   self.assertEqual(req.full_url,'https://api.openai.com/v1/responses');self.assertFalse(sent['store'])
   self.assertEqual(sent['reasoning'],{'effort':'low'});self.assertEqual(c.metrics[0]['usage']['input_tokens'],10)
 def test_credit_error_redacts_message(self):
  body={'error':{'code':'credit_balance_exhausted','message':'test-secret-not-a-real-key'}}
  error=HTTPError('https://api.openai.com/v1/responses',429,'Too many requests',{},io.BytesIO(json.dumps(body).encode()))
  with patch('llm_client.urlopen',side_effect=error):
   with self.assertRaises(ProviderError) as caught:LLMClient().create(input='test')
   self.assertIn('credit_balance_exhausted',str(caught.exception));self.assertNotIn('test-secret',str(caught.exception))
 def test_incomplete_is_not_success(self):
  with patch('llm_client.urlopen',return_value=io.BytesIO(b'{"status":"incomplete"}')):
   with self.assertRaises(ProviderError):LLMClient().create(input='test')
if __name__=='__main__':unittest.main()
