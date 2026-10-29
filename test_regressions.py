import copy,os,unittest
from pathlib import Path
from unittest.mock import patch
import retrieval as agent
class RegressionTests(unittest.TestCase):
 def setUp(self):
  env=patch.dict(os.environ,{'MOVIES_CSV':str(Path(__file__).with_name('movies.csv')),'SEARCH_MODE':'lexical'});env.start();self.addCleanup(env.stop)
 def test_hours(self):
  for text,expected in [('不超过两小时',120),('2小时内',120),('两个小时以内',120),('under 2 hours',119),('no more than 2 hours',120),('不超过一个半小时',90),('不超过1.5 hours',90),('不超过两小时半',150),('at least 2 hours',None)]:
   with self.subTest(text=text):
    c,_=agent.parse_constraints(text,{})
    if expected is None:self.assertEqual(c['min_runtime'],120)
    else:self.assertEqual(c['max_runtime'],expected);self.assertNotIn('min_runtime',c)
 def test_demo(self):
  r=agent.run_agent('想看轻松的科幻电影，最好不超过两小时',{})
  self.assertTrue(r['movies']);self.assertEqual(r['constraints']['max_runtime'],120)
  self.assertTrue(all(m['runtime']<=120 for m in r['movies']))
 def test_strict_boundary(self):
  m={'record_id':'x','runtime':120,'year':2020,'genres':['Sci-Fi']}
  self.assertFalse(agent.matches(m,agent.parse_constraints('under 2 hours')[0]))
  self.assertTrue(agent.matches(m,agent.parse_constraints('不超过两小时')[0]))
 def test_seen_state_no_mutation_reset(self):
  first=agent.run_agent('推荐一部科幻电影，不超过两小时',{})
  before=copy.deepcopy(first['constraints'])
  second=agent.run_agent('换一部',first['constraints'])
  self.assertTrue(second['movies']);self.assertEqual(first['constraints'],before)
  self.assertTrue(set(m['record_id'] for m in first['movies']).isdisjoint(m['record_id'] for m in second['movies']))
  self.assertEqual(second['constraints']['max_runtime'],120)
  self.assertEqual(agent.run_agent('推荐一部科幻电影，不超过两小时',{})['movies'],first['movies'])
  reset=agent.run_agent('重新开始，推荐一部科幻电影，不超过两小时',second['constraints'])
  self.assertEqual(reset['movies'],first['movies'])
 def test_exhausted(self):
  first=agent.run_agent('推荐六部电影',{'min_year':2026,'max_year':2026,'limit':6})
  self.assertEqual(agent.run_agent('换一部',first['constraints'])['movies'],[])
 def test_sequel_three_states(self):
  for status in ('true','false','unknown',''):
   m={'record_id':'x','runtime':90,'year':2020,'genres':['Sci-Fi'],'is_sequel':status,'sequel_evidence':'review'}
   self.assertEqual(agent.matches(m,{'exclude_sequels':True}),status=='false')
 def test_sequel_real_data_and_clear(self):
  r=agent.run_agent('科幻电影，不要续集',{})
  self.assertTrue(r['constraints']['exclude_sequels']);self.assertEqual(len(r['movies']),3)
  self.assertTrue(all('Non-sequel curation:' in m['evidence'] for m in r['movies']))
  self.assertNotIn('续集',r['constraints'].get('query_text',''))
  self.assertFalse(agent.parse_constraints('允许续集',r['constraints'])[0]['exclude_sequels'])
 def test_gore_explicit_and_clear(self):
  r=agent.run_agent('科幻电影，不要太血腥',{})
  self.assertTrue(r['constraints']['avoid_gore']);self.assertEqual(r['movies'],[])
  self.assertTrue(any('无法验证' in x for x in r['trace']))
  self.assertNotIn('血腥',r['constraints'].get('query_text',''))
  self.assertTrue(agent.run_agent('血腥程度不限',r['constraints'])['movies'])
 def test_state_validation(self):
  for bad in [{'seen_movie_ids':'x'},{'seen_movie_ids':[1]},{'exclude_sequels':'false'},{'avoid_gore':1}]:
   with self.assertRaises(ValueError):agent.validate_constraints(bad)
if __name__=='__main__':unittest.main()
