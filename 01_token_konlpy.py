import nltk, konlpy, time

text = "최성록 교수의 NLP 수업입니다. 수업은 오전 9:00에 시작하지 않지만, 약 1,000개의 예제,실습,그리고숙제가 있습니다."

tokenizers_word = [
    {'name': 'NLTK recommended',      'tokenizer': nltk.tokenize.word_tokenize},
    {'name': 'KoNLPy Hannanum',       'tokenizer': konlpy.tag.Hannanum().morphs},
    {'name': 'KoNLPy Kokoma',         'tokenizer': konlpy.tag.Kkma().morphs},
    {'name': 'KoNLPy Komoran',        'tokenizer': konlpy.tag.Komoran().morphs},
    {'name': 'KoNLPy OpenKoreanText', 'tokenizer': konlpy.tag.Okt().morphs},
    #{'name': 'KoNLPy MeCab-Ko',       'tokenizer': konlpy.tag.Mecab().morphs}, # Error in Windows
]

for tw in tokenizers_word:
    start = time.time()
    tokens = tw['tokenizer'](text)
    elapse = time.time() - start
    print(f"### {tw['name']} (time: {elapse:.3f} [sec])")
    print(tokens)
