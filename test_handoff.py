import os,unittest
from pathlib import Path
from unittest.mock import patch
import retrieval as agent
ROOT=Path(__file__).resolve().parent
class HandoffTests(unittest.TestCase):
 def setUp(self):
  self.env=patch.dict(os.environ,{'MOVIES_CSV':str(ROOT/'movies.csv'),'SEARCH_MODE':'lexical','OPENAI_API_KEY':'','OPENAI_MODEL':''});self.env.start();self.addCleanup(self.env.stop)
 def test_dataset(self):
  rows=agent.load_movies();self.assertEqual(len(rows),584)
  self.assertEqual(min(m['year'] for m in rows),1950);self.assertEqual(max(m['year'] for m in rows),2026)
  self.assertEqual(sum(bool(m.get('poster_url')) for m in rows),530)
 def test_frontend_contract(self):
  r=agent.run_agent('推荐三部2020年以后的科幻电影，不超过150分钟',{})
  self.assertEqual(set(r),{'constraints','movies','trace'});self.assertEqual(len(r['movies']),3)
  self.assertTrue(all(isinstance(m['genres'],str) for m in r['movies']))
  self.assertTrue(all(m['year']>2020 and m['runtime']<=150 for m in r['movies']))
 def test_posters_match_csv(self):
  r=agent.run_agent('推荐六部电影',{'min_year':2026,'max_year':2026,'limit':6})
  self.assertEqual(len(r['movies']),6)
  source={m['record_id']:m for m in agent.load_movies()}
  for m in r['movies']:
   self.assertTrue(m['poster_url'].startswith('https://upload.wikimedia.org/'))
   self.assertEqual(m['poster_url'],source[m['record_id']]['poster_url'])
   self.assertTrue(m['poster_source_url']);self.assertTrue(m['poster_license'])
 def test_multiturn(self):
  first=agent.run_agent('科幻电影，不超过150分钟',{'min_year':2020})
  result=agent.run_agent('新一点',first['constraints'])
  for k in ('genres','max_runtime','min_year'):self.assertEqual(result['constraints'][k],first['constraints'][k])
 def test_no_future_movies(self):
  self.assertEqual(agent.run_agent('movies after 2026',{})['movies'],[])
 def test_baseline_unconfigured(self):
  with self.assertRaises(agent.ConfigurationError):agent.run_baseline('Recommend movies')
if __name__=='__main__':unittest.main()
