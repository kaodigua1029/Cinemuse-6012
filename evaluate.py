"""Explicit live experiment runner. Writes actual results; never fabricates scores."""
import argparse,json,time
from pathlib import Path
from agent import run_agent,run_baseline,ConfigurationError,ProviderError,DataError

CASES=[
 {'id':'hours_zh','query':'推荐三部科幻电影，最好不超过两小时','expected':{'genres':['Sci-Fi'],'max_runtime':120}},
 {'id':'another','query':'换一部','expected':{'genres':['Sci-Fi'],'max_runtime':120,'limit':1},'followup':True},
 {'id':'quantity_reset','query':'再推荐几部','expected':{'genres':['Sci-Fi'],'max_runtime':120,'limit':3},'followup':True},
 {'id':'hours_en','query':'Recommend three sci-fi movies under 2 hours released after 2010.','expected':{'genres':['Sci-Fi'],'max_runtime':119,'min_year':2011}},
 {'id':'sequel','query':'推荐电影，不要续集','expected':{'exclude_sequels':True}},
 {'id':'gore','query':'不要太血腥的电影','expected':{'avoid_gore':True}},
]

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',default='live_results.json');p.add_argument('--cases',type=int,default=1);args=p.parse_args()
 if not 1<=args.cases<=len(CASES):p.error('--cases must be 1–6')
 results=[];state={};history=[];seen=set()
 for case in CASES[:args.cases]:
  if not case.get('followup'):state={};history=[];seen=set()
  history.append(case['query']);entry={'case':case,'status':'pending'}
  start=time.monotonic()
  try:
   a=run_agent(case['query'],state);entry['agent']=a;entry['agent_seconds']=round(time.monotonic()-start,3)
   ids=[m['record_id'] for m in a['movies']]
   entry['agent_state_matches_expected']=all(a['constraints'].get(k)==v for k,v in case['expected'].items())
   entry['agent_repeat_count']=len(set(ids)&seen)
   # Empty result is not treated as 100% recommendation accuracy.
   checks=[all((m['runtime']<=v if k=='max_runtime' else m['year']>=v if k=='min_year' else bool(set(v)&set(m['genres'].split(' / '))) if k=='genres' else True) for k,v in case['expected'].items()) for m in a['movies']]
   entry['agent_numeric_genre_satisfaction']=sum(checks)/len(checks) if checks else None
   entry['agent_returned_count']=len(ids);state=a['constraints'];seen.update(ids)
   prompt='以下是同一用户按顺序提出的要求。理解最后一轮并保留未取消的条件，直接推荐电影并给出年份、分钟时长和理由：\n'+'\n'.join(f'{i+1}. {q}' for i,q in enumerate(history))
   start=time.monotonic();entry['baseline']=run_baseline(prompt);entry['baseline_seconds']=round(time.monotonic()-start,3)
   entry['manual_review']={'baseline_constraints':None,'baseline_fact_accuracy':None,'agent_reason_support':None,'mood_match':None,'notes':''}
   entry['status']='completed'
  except (ConfigurationError,ProviderError,DataError,ValueError) as exc:
   entry['status']='failed';entry['error']=str(exc)
  results.append(entry)
  Path(args.output).write_text(json.dumps({'real_api_calls':True,'cases':results,'note':'仅为执行结果；人工评价项留空。数字/类型检查不验证剧情理解或全部事实。Baseline每轮是含完整用户要求的单次调用，不含检索资料和先前推荐历史，去重对比须单独披露。'},ensure_ascii=False,indent=2))
  print(case['id'],entry['status'])
  if entry['status']=='failed':break

if __name__=='__main__':main()
