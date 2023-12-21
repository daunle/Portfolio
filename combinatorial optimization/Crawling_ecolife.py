import pickle
import scrapy
url = 'https://ecolife.me.go.kr/ecolife/slfsfcfst/index?side=SM020202&page='
t = 33484 # 게시판 페이지 수 지정
idx = []
for i in tqdm(range(1,t+1)):
    ul = url + str(i)
    hdr = {'User-Agent':generate_user_agent(os='win',device_type='desktop')}
    request = requests.get(ul,headers = hdr)
    soup = bs(request.text, "lxml")
    a = soup.find_all('tr',attrs={'class':'hand list'}) #html 요소 설정
    for j in  a:
        idx.append(j['onclick'].replace("moveShow(","").replace(");","")) 

def count_old(dic,param):
    if param in dic.keys():
        dic[param] += 1
    else:
        dic[param] = 2

sa2 = {}
co = {}
for ix in tqdm(idx):
    url = 'https://ecolife.me.go.kr/ecolife/slfsfcfst/show?side=SM020202&seq='+ix # 크롤링 URL 설정
    hdr = {'User-Agent':generate_user_agent(os='win',device_type='desktop')} # User-Agent 설정 => 반복적인 크롤링시, 사이트에서 차단하는 경우 존재 이를 방지하기 위한 콛,
    request = requests.get(url,headers = hdr) 
    html = bs(request.text,'lxml')
    data = pd.read_html(url) # 사이트 속 table 추출
    if data[1][0][0] == '제품명':
        if data[1][1][0] in sa2.keys():
            for nb in range(2,100):
                if data[1][1][0]+f'_{nb}' in sa2.keys():
                    continue
                else:
                    sa2[data[1][1][0]+f'_{nb}'] = data
                    break
            count_old(dic = co,param = data[1][1][0])
        sa2[data[1][1][0]] = data
    else:
        sa2[ix] = data
    time.sleep(.1)
