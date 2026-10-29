import copy,json,unittest,os
from pathlib import Path
from unittest.mock import patch
import agent
import retrieval as db

class FakeLLM:
 def __init__(self):self.metrics=[];self.requests=[];self.fail_plan=False;self.fail_compose=False;self.always_bad=False;self.blocked=False
 def create(self,**kw):
  self.requests.append(kw);self.metrics.append({'model':'test-double','usage':{}})
  if 'tools' in kw:
   u=json.loads(kw['input'][0]['content']);c,_=db.parse_constraints(u['current'],u['previous'])
   args={'reset':bool('重新开始' in u['current']),'constraints':{k:c.get(k) for k in agent.PROPS},'unsupported':['streaming availability'] if self.blocked else []}
   if self.fail_plan:args['constraints']['max_runtime']='bad';self.fail_plan=False
   return {'status':'completed','output':[{'type':'function_call','name':'search_movies','call_id':'call'+str(len(self.requests)),'arguments':json.dumps(args)}]}
  if 'text' in kw:
   data=json.loads(next(x['output'] for x in reversed(kw['input']) if x.get('type')=='function_call_output'))
   recs=[{'record_id':x['record_id'],'reason':'根据提供的剧情推荐，情绪偏好仅供参考。','evidence_id':x['evidence_snippets'][0]['evidence_id']} for x in data['candidates'][:data['constraints']['limit']]]
   if self.fail_compose or self.always_bad:recs[0]['record_id']='hallucinated';self.fail_compose=False
   return {'status':'completed','output':[{'type':'message','role':'assistant','content':[{'type':'output_text','text':json.dumps({'recommendations':recs,'message':'基于检索证据。'})}]}]}
  return {'status':'completed','output':[{'type':'message','content':[{'type':'output_text','text':'baseline answer'}]}]}

class AgentTests(unittest.TestCase):
 def setUp(self):
  self.fake=FakeLLM();p=patch('agent.LLMClient',return_value=self.fake);p.start();self.addCleanup(p.stop)
  env=patch.dict(os.environ,{'MOVIES_CSV':str(Path(__file__).with_name('movies.csv')),'SEARCH_MODE':'lexical'});env.start();self.addCleanup(env.stop)
 def test_real_graph_with_fake_llm(self):
  r=agent.run_agent('科幻电影，不超过两小时',{})
  self.assertEqual(set(r),{'constraints','movies','trace'});self.assertEqual(len(r['movies']),3)
  self.assertTrue(all(m['runtime']<=120 and isinstance(m['genres'],str) for m in r['movies']))
  self.assertEqual(len(self.fake.requests),2);self.assertIn('tools',self.fake.requests[0])
 def test_multiturn_quantity_dedup(self):
  a=agent.run_agent('想看轻松的科幻电影，不超过两小时',{})
  old=copy.deepcopy(a['constraints']);b=agent.run_agent('换一部',a['constraints']);c=agent.run_agent('再推荐几部',b['constraints'])
  self.assertEqual(a['constraints'],old);self.assertEqual(len(b['movies']),1);self.assertEqual(len(c['movies']),3)
  ids=[m['record_id'] for r in (a,b,c) for m in r['movies']];self.assertEqual(len(ids),len(set(ids)))
  self.assertEqual(c['constraints']['max_runtime'],120)
 def test_baseline_no_tools_or_corpus(self):
  with patch('agent.db.load_movies',side_effect=AssertionError('must not read corpus')):
   self.assertEqual(agent.run_baseline('推荐电影'),'baseline answer')
  self.assertEqual(len(self.fake.requests),1);self.assertEqual(self.fake.requests[0],{'input':'推荐电影'})
 def test_gore_fails_closed(self):
  r=agent.run_agent('不要太血腥',{});self.assertEqual(r['movies'],[]);self.assertEqual(len(self.fake.requests),1)
 def test_unsupported_fails_closed(self):
  self.fake.blocked=True;r=agent.run_agent('Netflix available',{});self.assertEqual(r['movies'],[])
  self.assertEqual(r['constraints']['unsupported_constraints'],['streaming availability'])
 def test_sequels_curated(self):
  r=agent.run_agent('不要续集',{});self.assertEqual(len(r['movies']),3)
  self.assertTrue(all('Non-sequel review' in m['evidence'] for m in r['movies']))
 def test_plan_repair(self):
  self.fake.fail_plan=True;r=agent.run_agent('科幻',{});self.assertTrue(r['movies']);self.assertEqual(len(self.fake.requests),3)
 def test_hallucinated_id_repair(self):
  self.fake.fail_compose=True;r=agent.run_agent('科幻',{});self.assertTrue(r['movies']);self.assertEqual(len(self.fake.requests),3)
 def test_bounded_failure_no_fallback(self):
  self.fake.always_bad=True
  with self.assertRaises(agent.ProviderError):agent.run_agent('科幻',{})
  self.assertEqual(len(self.fake.requests),3)
 def test_guard_numeric_hours(self):
  args={'reset':False,'unsupported':[],'constraints':{k:None for k in agent.PROPS}}
  args['constraints']['max_runtime']=190
  c,_=agent.validate_plan(args,{},'under 2 hours');self.assertEqual(c['max_runtime'],119)
 def test_missing_key_fails(self):
  from llm_client import LLMClient
  with patch.dict(os.environ,{'OPENAI_API_KEY':'','MOVIE_ENV_FILE':'/nonexistent-config'}):
   with self.assertRaises(agent.ConfigurationError):LLMClient()
 def test_evidence_snippets_are_source_text(self):
  for m in db.load_movies()[:10]:
   snippets=agent.evidence_snippets(m)
   self.assertTrue(snippets)
   for item in snippets:self.assertIn(item['text'],m['overview']);self.assertGreaterEqual(len(item['text']),20)
 def test_quote_whitespace_preserves_source(self):
  source='Abe reasons that they have created a time machine.  After some testing, Abe builds it.'
  quote='Abe reasons that they have created a time machine. After some testing'
  self.assertEqual(agent.source_quote(quote,source),'Abe reasons that they have created a time machine.  After some testing')
  self.assertIsNone(agent.source_quote(quote.replace('time machine','space machine'),source))
 def test_fake_quote_rejected(self):
  row=db.load_movies()[0];c={'limit':3};state={'constraints':c,'candidates':[row],'draft':{'recommendations':[{'record_id':row['record_id'],'reason':'x','evidence_id':'fabricated-id'}],'message':''},'compose_attempts':2}
  with self.assertRaises(agent.ProviderError):agent.verify(state)
if __name__=='__main__':unittest.main()

