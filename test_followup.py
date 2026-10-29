import copy,unittest
import os
from pathlib import Path
from unittest.mock import patch
import retrieval as agent
class FollowupTests(unittest.TestCase):
 def setUp(self):
  env=patch.dict(os.environ,{'MOVIES_CSV':str(Path(__file__).with_name('movies.csv')),'SEARCH_MODE':'lexical'});env.start();self.addCleanup(env.stop)
 def test_quantity_is_per_turn(self):
  first=agent.run_agent('科幻电影，不超过两小时',{})
  second=agent.run_agent('换一部',first['constraints'])
  previous=copy.deepcopy(second['constraints'])
  third=agent.run_agent('再推荐几部',second['constraints'])
  self.assertEqual(len(second['movies']),1)
  self.assertEqual(third['constraints']['limit'],3)
  self.assertEqual(len(third['movies']),3)
  self.assertEqual(second['constraints'],previous)
  for k in ('genres','max_runtime'):self.assertEqual(third['constraints'][k],first['constraints'][k])
  ids=[m['record_id'] for r in (first,second,third) for m in r['movies']]
  self.assertEqual(len(ids),len(set(ids)))
 def test_explicit_count_overrides(self):
  for query,n in [('推荐五部',5),('recommend two movies',2),('换一部',1),('再推荐几部',3),('新一点',3)]:
   with self.subTest(query=query):self.assertEqual(agent.parse_constraints(query,{'limit':1})[0]['limit'],n)
 def test_topic_cleanup(self):
  for query in ('想看轻松的科幻电影','我想看轻松的科幻电影','请帮我推荐轻松的科幻电影'):
   with self.subTest(query=query):self.assertEqual(agent.parse_constraints(query,{})[0]['query_text'],'轻松')
  self.assertNotIn('query_text',agent.parse_constraints('再推荐几部',{})[0])
 def test_old_topic_state(self):
  old={'query_text':'想看轻松','limit':1,'genres':['Sci-Fi']}
  new,_=agent.parse_constraints('再推荐几部',old)
  self.assertEqual(new['query_text'],'轻松');self.assertEqual(old['query_text'],'想看轻松')
 def test_meaningful_topic_preserved(self):
  c,_=agent.parse_constraints('想看关于自我意识的电影',{})
  self.assertIn('自我意识',c['query_text'])
if __name__=='__main__':unittest.main()
