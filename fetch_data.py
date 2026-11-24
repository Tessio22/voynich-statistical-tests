#!/usr/bin/env python3
"""Download every corpus used by the experiments into ./data, straight from the original sources.

Nothing here is redistributed in the repository: the Voynich transliteration belongs to René Zandbergen
(voynich.nu), the comparison texts come from Wikisource / Wikipedia (CC BY-SA), and the Lingua Ignota
word list is read from Philip Neal's page. Run:  python3 fetch_data.py            (everything, ~15-30 min)
                                              python3 fetch_data.py --only voynich,latin,italian,german
"""
import argparse, html, json, os, re, sys, time, urllib.parse, urllib.request

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
UA = {'User-Agent': 'voynich-statistical-tests/1.0 (research script; contact via GitHub)'}

def get(url, tries=5):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read().decode('utf-8', 'replace')
        except Exception as e:
            wait = 5 * (k + 1)
            print(f'   retry in {wait}s ({e})', file=sys.stderr); time.sleep(wait)
    raise RuntimeError('failed: ' + url)

def html_to_text(h):
    for tag in ('table', 'style', 'script', 'sup'):
        h = re.sub(rf'<{tag}\b.*?</{tag}>', ' ', h, flags=re.S | re.I)
    h = re.sub(r'<span class="mw-editsection.*?</span>\s*</span>', ' ', h, flags=re.S)
    h = re.sub(r'<br\s*/?>|</p>|</div>|</li>|</h\d>', '\n', h, flags=re.I)
    h = re.sub(r'<[^>]+>', '', h)
    return html.unescape(h)

def wiki_page(host, title):
    url = f'https://{host}/w/api.php?action=parse&format=json&prop=text&page=' + urllib.parse.quote(title)
    for k in range(4):
        try:
            j = json.loads(get(url))
            return html_to_text(j['parse']['text']['*']) if 'parse' in j else ''
        except json.JSONDecodeError:
            time.sleep(5)
    return ''

def wiki_prefix(host, prefix, limit=200):
    url = f'https://{host}/w/api.php?action=query&list=allpages&format=json&aplimit={limit}&apprefix=' + urllib.parse.quote(prefix)
    return [p['title'] for p in json.loads(get(url))['query']['allpages']]

def wiki_category(host, cat):
    url = f'https://{host}/w/api.php?action=query&list=categorymembers&cmnamespace=0&cmlimit=500&format=json&cmtitle=' + urllib.parse.quote(cat)
    return [p['title'] for p in json.loads(get(url))['query']['categorymembers']]

def pages(host, titles, sleep=0.6):
    out = []
    for i, t in enumerate(titles):
        out.append(wiki_page(host, t)); time.sleep(sleep)
        if (i + 1) % 10 == 0: print(f'   {i+1}/{len(titles)}')
    return '\n\n'.join(out)

ROMAN = 'I II III IV V VI VII VIII IX X XI XII XIII XIV XV XVI XVII XVIII XIX XX XXI XXII XXIII XXIV XXV XXVI XXVII XXVIII XXIX XXX XXXI XXXII XXXIII XXXIV XXXV XXXVI XXXVII'.split()
ORD = ['prima', 'seconda', 'terza', 'quarta', 'quinta', 'sesta', 'settima', 'ottava', 'nona', 'decima']

def heb_num(n):
    ones = ['', 'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט']; tens = ['', 'י', 'כ', 'ל', 'מ', 'נ']
    return {15: 'טו', 16: 'טז'}.get(n, tens[n // 10] + ones[n % 10])

def f_voynich():
    t = get('https://www.voynich.nu/data/ZL3b-n.txt')
    open(os.path.join(DATA, 'ZL3b.txt'), 'w', encoding='latin-1', errors='replace').write(t)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
    from parse import parse
    json.dump(parse(path=os.path.join(DATA, 'ZL3b.txt'), comma='split'), open(os.path.join(DATA, 'voynich_lines.json'), 'w'))
    return None

SOURCES = {
 'voynich':  ('ZL3b.txt', f_voynich),
 'latin':    ('latin.txt', lambda: pages('la.wikisource.org', ['De viribus herbarum', 'Commentarii de bello Gallico/Liber I', 'Commentarii de bello Gallico/Liber V'])),
 'pliny':    ('pliny.txt', lambda: pages('la.wikisource.org', ['Naturalis Historia/Liber ' + r for r in ['XX', 'XXI', 'XXII', 'XXIII', 'XXIV', 'XXV', 'XXVI', 'XXVII', 'XIX', 'XVIII']])),
 'latin_big':('latin_big.txt', lambda: pages('la.wikisource.org',
               ['De medicina (Celsus)/' + r for r in ROMAN[:8]] + ['Etymologiae (Isidorus)'] +
               ['De re rustica/Liber ' + r for r in ROMAN[:12]] +
               [('Naturalis historia/Liber ' if r == 'I' else 'Naturalis Historia/Liber ') + r for r in ROMAN[:17] + ROMAN[27:37]])),
 'italian':  ('italian.txt', lambda: pages('it.wikisource.org', [f'Decameron/Giornata {g}/Novella {n}' for g in ('prima', 'seconda') for n in ORD])),
 'dante':    ('dante.txt', lambda: pages('it.wikisource.org', ['Divina Commedia/Inferno/Canto ' + r for r in ROMAN[:12]])),
 'german':   ('german.txt', lambda: pages('de.wikisource.org', ['Der Ackermann aus Böhmen (Handschrift 14. Jh.)'])),
 'czech':    ('czech.txt', lambda: pages('cs.wikisource.org', ['Bible kralická (1613)/Genesis'] + wiki_prefix('cs.wikisource.org', 'Dalimil/'))),
 'catalan':  ('catalan.txt', lambda: pages('ca.wikisource.org', wiki_prefix('ca.wikisource.org', 'Tirant lo Blanch (1905)/1/')[:70] + wiki_prefix('ca.wikisource.org', 'Crònica de Bernat Desclot/Capítol')[:80])),
 'occitan':  ('occitan.txt', lambda: pages('wikisource.org', wiki_category('wikisource.org', 'Category:Occitan'))),
 'hebrew':   ('hebrew.txt', lambda: pages('he.wikisource.org', ['בראשית ' + heb_num(i) for i in range(1, 51)] + ['גור אריה על רש"י בראשית א'])),
 'arabic':   ('arabic.txt', lambda: pages('ar.wikisource.org', ['القانون في الطب - الكتاب الأول'])),
 'persian':  ('persian.txt', lambda: pages('fa.wikisource.org', ['کلیات سعدی/گلستان/' + b for b in ['دیباچه', 'باب اول', 'باب دوم', 'باب سوم', 'باب چهارم', 'باب پنجم', 'باب ششم', 'باب هفتم', 'باب هشتم']] +
               ['تاریخ بیهقی/' + b for b in ['کارهای مسعود بعد از وفات محمود', 'بقیهٔ احوال امیر محمد', 'کارهای هرات', 'آغاز تاریخ مسعود (خطبه)', 'نامهٔ تکیناباد']])),
 'turkish':  ('turkish.txt', lambda: pages('tr.wikisource.org', wiki_prefix('tr.wikisource.org', 'Kitâb-ı Dedem Korkut', 100) + wiki_prefix('tr.wikisource.org', 'Hikayet-i Oğuznâme', 100) + ["Dîvânü Lugâti't-Türk"])),
 'nahuatl':  ('nahuatl.txt', None),
 'lingua_ignota': ('lingua_ignota.txt', None),
}

def f_nahuatl():
    url = 'https://nah.wikipedia.org/w/api.php?action=query&list=allpages&aplimit=500&apminsize=6000&format=json'
    titles = [p['title'] for p in json.loads(get(url))['query']['allpages']][:250]
    out = []
    for t in titles:
        j = json.loads(get('https://nah.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&format=json&titles=' + urllib.parse.quote(t)))
        out.append(next(iter(j['query']['pages'].values())).get('extract', '')); time.sleep(0.2)
    return '\n\n'.join(out)

def f_lingua_ignota():
    h = get('http://philipneal.net/voynichsources/lingua_ignota_hildegard/')
    words = []
    for row in re.findall(r'<tr>(.*?)</tr>', h, flags=re.S):
        cells = re.findall(r'<td[^>]*>(.*?)</td>', row, flags=re.S)
        if len(cells) >= 2:
            w = re.sub(r'[^a-z]', '', html.unescape(re.sub(r'<[^>]+>', '', cells[0])).lower())
            if len(w) > 1: words.append(w)
    return '\n'.join(dict.fromkeys(words))

SOURCES['nahuatl'] = ('nahuatl.txt', f_nahuatl)
SOURCES['lingua_ignota'] = ('lingua_ignota.txt', f_lingua_ignota)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--only', help='comma-separated keys: ' + ','.join(SOURCES))
    a = ap.parse_args(); os.makedirs(DATA, exist_ok=True)
    keys = a.only.split(',') if a.only else list(SOURCES)
    for k in keys:
        fname, fn = SOURCES[k]; path = os.path.join(DATA, fname)
        if os.path.exists(path) and os.path.getsize(path) > 1000:
            print(f'= {k}: already present'); continue
        print(f'> {k} -> data/{fname}')
        txt = fn()
        if txt is not None: open(path, 'w', encoding='utf-8').write(txt)
        print(f'  done, {os.path.getsize(path)//1024} KB')
