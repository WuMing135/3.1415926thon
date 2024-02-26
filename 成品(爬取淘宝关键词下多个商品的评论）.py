import time
import re
import pandas as pd
import hashlib
from fake_useragent import UserAgent
import requests
def get_random_user_agent():
    user_agent = UserAgent()
    return user_agent.random
def hex_md5(s):
    m = hashlib.md5()
    m.update(s.encode('UTF-8'))
    return m.hexdigest()
def get(item_id):
    sgcookie = 'E100z%2FVVDVIiGsoHPNce%2BiPz5IOGZIPgoeDq0aLLp%2F9azu4qhufI5SFz2mJwKiExTYfe37o2Twcy7fq0PI6BOz7eS%2FWr45ATq3yuo99MdVdBVSk%3D'#不变值
    x5sec = '7b22733b32223a2264353866356337613765323063323939222c22617365727665723b33223a22307c43506d6f30363047454a58382f666342476841794d6a45334d6a67334d5449794f5455334f7a45784d4a655a354e7346227d'#每次运行前需要更换
    pat = re.compile('"reviewWordContent":"(.*?)","skuText"')
    url = 'https://h5api.m.taobao.com/h5/mtop.alibaba.review.list.for.new.pc.detail/1.0/'
    appKey = '12574478'#不变值
    # 获取当前时间戳
    t = str(int(time.time() * 1000))
    data = '{"itemId":' + str(item_id) + ',"bizCode":"ali.china.tmall","channel":"pc_detail","pageSize":20,"pageNum":1}'
    # 原始数据字符串
    #data_str = "{\"id\":\"43801547538\",\"detail_v\":\"3.3.2\",\"exParams\":\"{\\\"ali_refid\\\":\\\"a3_430582_1006:1103998642:N:H2T4xOzf7HBDj7r Ma9/9Q==:76e07dfc47faef8a374f1ed1a8e005de\\\",\\\"ali_trackid\\\":\\\"1_76e07dfc47faef8a374f1ed1a8e005de\\\",\\\"id\\\":\\\"43801547538\\\",\\\"spm\\\":\\\"a21n57.1.0.0\\\",\\\"queryParams\\\":\\\"ali_refid=a3_430582_1006%3A1103998642%3AN%3AH2T4xOzf7HBDj7r%20Ma9%2F9Q%3D%3D%3A76e07dfc47faef8a374f1ed1a8e005de&ali_trackid=1_76e07dfc47faef8a374f1ed1a8e005de&id=43801547538&spm=a21n57.1.0.0\\\",\\\"domain\\\":\\\"https://item.taobao.com\\\",\\\"path_name\\\":\\\"/item.htm\\\"}\"}"
    #data = data_str.replace("43801547538", item_id)# 使用字符串替换将 "43801547538" 替换为变量值
    cookies = {
        'sgcookie': sgcookie,
        'x5sec': x5sec,
    }
    headers = {
        'referer': 'https://item.taobao.com/',
        'user-agent': get_random_user_agent(),
    }
    params = {
        'appKey': appKey,
        'data': data,
        't': t,
    }
    # 请求空获取cookies
    response = requests.get(url=url, params=params, cookies=cookies, headers=headers)
    Set_Cookie =response.headers['Set-Cookie']
    # 使用正则表达式从Set-Cookie字符串中提取_m_h5_tk和_m_h5_tk_enc的值
    m_h5_tk_match = re.search(r'_m_h5_tk=([^;]+)', Set_Cookie)
    _m_h5_tk = m_h5_tk_match.group(1) if (m_h5_tk_match) else None
    m_h5_tk_enc_match = re.search(r'_m_h5_tk_enc=([^;]+)', Set_Cookie)
    _m_h5_tk_enc = m_h5_tk_enc_match.group(1) if m_h5_tk_enc_match else None
    # 查看响应状态码
    #print("Status Code:", response.status_code)
    # 查看响应头
    #print("Headers:", response.headers)
    # 查看响应正文
    #print("Content:", response.text)
    #print(Set_Cookie)
    #print("_m_h5_tk:", _m_h5_tk)
    #print("_m_h5_tk_enc:", _m_h5_tk_enc)
    token = _m_h5_tk.split('_')[0]
    u = token + '&' + t + '&' + appKey + '&' + data
    # MD5加密
    sign = hex_md5(u)
    #print('秘钥：' + sign)
    # 设置第二次请求的cookie
    headers = {
        'Referer': 'https://item.taobao.com/',
        'User-Agent': get_random_user_agent(),
    }
    cookies = {
        'sgcookie': sgcookie,
        '_m_h5_tk': _m_h5_tk,
        '_m_h5_tk_enc': _m_h5_tk_enc,
        'x5sec': x5sec,
    }
    params = {
        'appKey': appKey,
        't': t,
        'sign': sign,
        'data': data,
    }
    # 发送请求
    response = requests.get(url=url, params=params, cookies=cookies, headers=headers)
    print(response.text)
    response.raise_for_status()  # 检查请求是否成功
    # 提取评论
    comments = pat.findall(response.text)
    texts.extend(comments)
    print(f'爬取成功，评论数量：{len(comments)}')
    response = requests.get(url)
    # 获取返回状态码
    status_code = response.status_code
    print(f"Status Code: {status_code}")
    # 休眠一段时间，以避免请求频率过快
    time.sleep(3)
#直接用curl进行转换，然后再params中删去三行带有jsonp的代码，并且将useragent换掉
cookies = {
    'cna': 'NK5PHdSLJwgBASQJiiC1gj93',
    'miid': '258111831191843291',
    'thw': 'cn',
    't': '846c99e0c0b8339c3031ca81bb0f1bab',
    'mt': 'ci=0_1',
    'xlly_s': '1',
    'lgc': 'tb890178762166',
    'tracknick': 'tb890178762166',
    '_samesite_flag_': 'true',
    'cookie2': '16206db95ac5bdc63d6133595be0720c',
    '_tb_token_': '387531f56d433',
    '3PcFlag': '1706361866584',
    'sgcookie': 'E100iZx04fMZ4GBhg3O9MIQ%2FAk3BG2EnpHjH8V%2Brbi3taQ1z7inm20z61cluY%2BOwQbZQsSXbT6YdpwlaOtdjOwXP4y0600MYKouguoe0b2q2%2Fsk%3D',
    'unb': '2217287122957',
    'uc1': 'pas=0&existShop=false&cookie14=UoYekxnLTOnI9Q%3D%3D&cookie16=URm48syIJ1yk0MX2J7mAAEhTuw%3D%3D&cookie15=VFC%2FuZ9ayeYq2g%3D%3D&cookie21=URm48syIZx9a',
    'uc3': 'id2=UUpgQcD1YtjUTUsawA%3D%3D&lg2=VT5L2FSpMGV7TQ%3D%3D&nk2=F5RNbBgpUmhZ1VuMSPY%3D&vt3=F8dD3ChNXtKxa%2BU5h0Y%3D',
    'csg': '45e747b8',
    'cancelledSubSites': 'empty',
    'cookie17': 'UUpgQcD1YtjUTUsawA%3D%3D',
    'dnk': 'tb890178762166',
    'skt': 'e6ca4a7c522ca58f',
    'existShop': 'MTcwNjM2MTg5MQ%3D%3D',
    'uc4': 'id4=0%40U2gqztnVbqN8r8csKQqwgFCCkQhW1tm5&nk4=0%40FY4Gu6zBMkg0xZdupEavWPw1UzC1qkfJGg%3D%3D',
    '_cc_': 'VFC%2FuZ9ajQ%3D%3D',
    '_l_g_': 'Ug%3D%3D',
    'sg': '67a',
    '_nk_': 'tb890178762166',
    'cookie1': 'BxUGOlj0KhRN2AycLScC7jAY5W9X6gMVaP3c8H5LZjY%3D',
    'mtop_partitioned_detect': '1',
    '_m_h5_tk': '4df5142be9d65f402f0d376f1dff66e1_1706370174343',
    '_m_h5_tk_enc': '744594a2f716b52e956ba8ba19686936',
    'tfstk': 'eP26nHNgfFY62nRMRNsUAPYaaNDblP6y5niYqopwDAHtkWZulVScsAPIlyUIWdSGsj3bSAh0_quZlxZ0PM7PUTrgjxDdzaWrslG2NxB1868UjlDDxH8AaWEMhgk8SiQACi-VFkQBQdvY0v-wMTzURJis96rtJPUG0DG1hlgpHZySfE38X29vHgkkU46EqItIZKiIzMsBiIVi3wn4G2GwXfnnx3SCAQRm6DmIzMsBiIctxDXPAMOyi',
    'isg': 'BPHxrajBMDV7RZ1cDA2bYe7zAH2L3mVQTG-YcNMG6rjX-hFMGyzfINkcHI6cNP2I',
}

headers = {
    'authority': 'h5api.m.taobao.com',
    'accept': '*/*',
    'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
    # Requests sorts cookies= alphabetically
    # 'cookie': 'cna=NK5PHdSLJwgBASQJiiC1gj93; miid=258111831191843291; thw=cn; t=846c99e0c0b8339c3031ca81bb0f1bab; mt=ci=0_1; xlly_s=1; lgc=tb890178762166; tracknick=tb890178762166; _samesite_flag_=true; cookie2=16206db95ac5bdc63d6133595be0720c; _tb_token_=387531f56d433; 3PcFlag=1706361866584; sgcookie=E100iZx04fMZ4GBhg3O9MIQ%2FAk3BG2EnpHjH8V%2Brbi3taQ1z7inm20z61cluY%2BOwQbZQsSXbT6YdpwlaOtdjOwXP4y0600MYKouguoe0b2q2%2Fsk%3D; unb=2217287122957; uc1=pas=0&existShop=false&cookie14=UoYekxnLTOnI9Q%3D%3D&cookie16=URm48syIJ1yk0MX2J7mAAEhTuw%3D%3D&cookie15=VFC%2FuZ9ayeYq2g%3D%3D&cookie21=URm48syIZx9a; uc3=id2=UUpgQcD1YtjUTUsawA%3D%3D&lg2=VT5L2FSpMGV7TQ%3D%3D&nk2=F5RNbBgpUmhZ1VuMSPY%3D&vt3=F8dD3ChNXtKxa%2BU5h0Y%3D; csg=45e747b8; cancelledSubSites=empty; cookie17=UUpgQcD1YtjUTUsawA%3D%3D; dnk=tb890178762166; skt=e6ca4a7c522ca58f; existShop=MTcwNjM2MTg5MQ%3D%3D; uc4=id4=0%40U2gqztnVbqN8r8csKQqwgFCCkQhW1tm5&nk4=0%40FY4Gu6zBMkg0xZdupEavWPw1UzC1qkfJGg%3D%3D; _cc_=VFC%2FuZ9ajQ%3D%3D; _l_g_=Ug%3D%3D; sg=67a; _nk_=tb890178762166; cookie1=BxUGOlj0KhRN2AycLScC7jAY5W9X6gMVaP3c8H5LZjY%3D; mtop_partitioned_detect=1; _m_h5_tk=4df5142be9d65f402f0d376f1dff66e1_1706370174343; _m_h5_tk_enc=744594a2f716b52e956ba8ba19686936; tfstk=eP26nHNgfFY62nRMRNsUAPYaaNDblP6y5niYqopwDAHtkWZulVScsAPIlyUIWdSGsj3bSAh0_quZlxZ0PM7PUTrgjxDdzaWrslG2NxB1868UjlDDxH8AaWEMhgk8SiQACi-VFkQBQdvY0v-wMTzURJis96rtJPUG0DG1hlgpHZySfE38X29vHgkkU46EqItIZKiIzMsBiIVi3wn4G2GwXfnnx3SCAQRm6DmIzMsBiIctxDXPAMOyi; isg=BPHxrajBMDV7RZ1cDA2bYe7zAH2L3mVQTG-YcNMG6rjX-hFMGyzfINkcHI6cNP2I',
    'referer': 'https://s.taobao.com/',
    'sec-ch-ua': '"Not A(Brand";v="99", "Microsoft Edge";v="121", "Chromium";v="121"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'script',
    'sec-fetch-mode': 'no-cors',
    'sec-fetch-site': 'same-site',
    'user-agent': get_random_user_agent(),
}

params = {
    'jsv': '2.6.2',
    'appKey': '12574478',
    't': '1706361927369',
    'sign': 'aad6aee6846c18c379f364d580b0439f',
    'api': 'mtop.relationrecommend.WirelessRecommend.recommend',
    'v': '2.0',
    'data': '{"appId":"34385","params":"{\\"device\\":\\"HMA-AL00\\",\\"isBeta\\":\\"false\\",\\"grayHair\\":\\"false\\",\\"from\\":\\"nt_history\\",\\"brand\\":\\"HUAWEI\\",\\"info\\":\\"wifi\\",\\"index\\":\\"4\\",\\"rainbow\\":\\"\\",\\"schemaType\\":\\"auction\\",\\"elderHome\\":\\"false\\",\\"isEnterSrpSearch\\":\\"true\\",\\"newSearch\\":\\"false\\",\\"network\\":\\"wifi\\",\\"subtype\\":\\"\\",\\"hasPreposeFilter\\":\\"false\\",\\"prepositionVersion\\":\\"v2\\",\\"client_os\\":\\"Android\\",\\"gpsEnabled\\":\\"false\\",\\"searchDoorFrom\\":\\"srp\\",\\"debug_rerankNewOpenCard\\":\\"false\\",\\"homePageVersion\\":\\"v7\\",\\"searchElderHomeOpen\\":\\"false\\",\\"search_action\\":\\"initiative\\",\\"sugg\\":\\"_4_1\\",\\"sversion\\":\\"13.6\\",\\"style\\":\\"list\\",\\"ttid\\":\\"600000@taobao_pc_10.7.0\\",\\"needTabs\\":\\"true\\",\\"areaCode\\":\\"CN\\",\\"vm\\":\\"nw\\",\\"countryNum\\":\\"156\\",\\"m\\":\\"pc\\",\\"page\\":1,\\"n\\":48,\\"q\\":\\"%E9%9D%A2%E7%BA%B8\\",\\"tab\\":\\"all\\",\\"pageSize\\":48,\\"totalPage\\":100,\\"totalResults\\":4800,\\"sourceS\\":\\"0\\",\\"sort\\":\\"_coefp\\",\\"bcoffset\\":\\"\\",\\"ntoffset\\":\\"\\",\\"filterTag\\":\\"\\",\\"service\\":\\"\\",\\"prop\\":\\"\\",\\"loc\\":\\"\\",\\"start_price\\":null,\\"end_price\\":null,\\"startPrice\\":null,\\"endPrice\\":null,\\"itemIds\\":null,\\"p4pIds\\":null}"}',
}

response = requests.get('https://h5api.m.taobao.com/h5/mtop.relationrecommend.wirelessrecommend.recommend/2.0/', params=params, cookies=cookies, headers=headers)
texts = []
json_date = response.json()
auctions = json_date['data']['itemsArray']
for auction in auctions:
    try:
        item_id = auction['item_id']
        title = auction['title']
        print(item_id,title)
        get(item_id)
    except Exception as e:
        print('爬取失败：{e}')
        break  # 中断循环，停止爬取
# 将评论保存到Excel文件
df = pd.DataFrame({'评论': texts})
df.to_excel('评论数据.xlsx', index=False)
print(f'数据保存成功，总评论数量：{len(texts)}')
"""
    response = requests.get(url)
    # 查看响应中的 cookies
    cookies = response.cookies
    # 输出 cookies 信息
    print(cookies)
"""