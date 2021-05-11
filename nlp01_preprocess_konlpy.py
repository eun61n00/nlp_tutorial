import konlpy, time

text = "최성록 교수의 NLP 수업입니다. 수업은 오전 9:00에 시작하지 않지만, 약 1,000개의 예제,실습,그리고숙제가 있습니다."

pos_taggers = [
    {'name': 'KoNLPy Hannanum',       'tagger': konlpy.tag.Hannanum()},
    {'name': 'KoNLPy Kokoma',         'tagger': konlpy.tag.Kkma()},
    {'name': 'KoNLPy Komoran',        'tagger': konlpy.tag.Komoran()},
    {'name': 'KoNLPy OpenKoreanText', 'tagger': konlpy.tag.Okt()},
    #{'name': 'KoNLPy MeCab-Ko',       'tagger': konlpy.tag.Mecab()}, # Error in Windows
]

for pt in pos_taggers:
    start = time.time()
    pos_tag = pt['tagger'].pos(text) # cf. 'morphs' function
    elapse = time.time() - start
    print(f"### {pt['name']} (time: {elapse:.3f} [sec])")
    print(pos_tag)
