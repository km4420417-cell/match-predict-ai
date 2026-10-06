import os, math, statistics
from datetime import datetime, timezone
from typing import Optional
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
app=FastAPI(title="MATCH PREDICT AI",version="1.1")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])
API_KEY=os.getenv("API_FOOTBALL_KEY",""); BASE="https://v3.football.api-sports.io"
class PredictRequest(BaseModel):
    home_team:str; away_team:str; league_id:Optional[int]=None; season:Optional[int]=None
async def af_get(path,params):
    if not API_KEY: raise HTTPException(503,"API_FOOTBALL_KEY no configurada")
    async with httpx.AsyncClient(timeout=15) as c:
        r=await c.get(BASE+path,params=params,headers={"x-apisports-key":API_KEY}); r.raise_for_status()
        d=r.json()
        if d.get("errors"): raise HTTPException(502,str(d["errors"]))
        return d.get("response",[])
async def team_search(name): return await af_get("/teams",{"search":name})
def poisson(k,lam): return math.exp(-lam)*lam**k/math.factorial(k)
def normalize(v):
    s=sum(v); return [x/s for x in v] if s else [1/3]*len(v)
def p1x2(h,a):
    r=[0,0,0]
    for x in range(8):
        for y in range(8):
            p=poisson(x,h)*poisson(y,a); r[0 if x>y else 1 if x==y else 2]+=p
    return normalize(r)
@app.get("/health")
async def health(): return {"ok":True,"api_configured":bool(API_KEY),"utc":datetime.now(timezone.utc).isoformat()}
@app.post("/api/predict")
async def predict(req:PredictRequest):
    hs=await team_search(req.home_team); aws=await team_search(req.away_team)
    if not hs or not aws: raise HTTPException(404,"No se ha encontrado uno de los equipos.")
    home,away=hs[0]["team"],aws[0]["team"]
    hm=await af_get("/fixtures",{"team":home["id"],"last":10}); am=await af_get("/fixtures",{"team":away["id"],"last":10})
    def form(ms,tid):
        gf=[];ga=[]
        for m in ms:
            if m.get("fixture",{}).get("status",{}).get("short") not in ("FT","AET","PEN"): continue
            t,g=m.get("teams",{}),m.get("goals",{})
            if t.get("home",{}).get("id")==tid: gf.append(g.get("home") or 0);ga.append(g.get("away") or 0)
            else: gf.append(g.get("away") or 0);ga.append(g.get("home") or 0)
        return (statistics.mean(gf) if gf else 1.25,statistics.mean(ga) if ga else 1.10,len(gf))
    hgf,hga,hn=form(hm,home["id"]); agf,aga,an=form(am,away["id"])
    xh=max(.25,min(4,(hgf+aga)*.55+.25)); xa=max(.2,min(3.5,(agf+hga)*.45)); pr=p1x2(xh,xa)
    best=max(((x,y,poisson(x,xh)*poisson(y,xa)) for x in range(6) for y in range(6)),key=lambda z:z[2])
    q=min(1,(hn+an)/20)
    return {"source":"API-Football","updated_at":datetime.now(timezone.utc).isoformat(),"home":home["name"],"away":away["name"],"probabilities":{"home":round(pr[0]*100,1),"draw":round(pr[1]*100,1),"away":round(pr[2]*100,1)},"expected_goals":{"home":round(xh,2),"away":round(xa,2)},"predicted_score":f"{best[0]}-{best[1]}","confidence":round(5+4*q*max(pr),1),"data_quality":round(q*100),"factors":[f"{hn} partidos recientes de {home['name']}.",f"{an} partidos recientes de {away['name']}.",f"Goles a favor: {hgf:.2f} / {agf:.2f}.",f"Goles recibidos: {hga:.2f} / {aga:.2f}.","Modelo base: Poisson."]}
