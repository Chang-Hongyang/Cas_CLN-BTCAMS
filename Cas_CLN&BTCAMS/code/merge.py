import codecs
import json
from collections import Counter


def spo_to_text(spo):
    subject = spo['subject']
    predicate = spo['predicate']
    object = spo['object']['@value']

    return '$$'.join([subject, predicate, object])


def data_preprocess(dir_path):
    total_files = []

    files_num = 0
    null_num = 0

    for i in list(range(1, 60)):
        files_num += 1
        with open(dir_path + 'test_{}.json'.format(i), 'r')  as fr:
            print(dir_path + 'test_{}.json'.format(i))
            total_files.append(fr.readlines())
    with open(dir_path + 'test_{}.json'.format(60), 'r') as fr:
        print(dir_path + 'test_{}.json'.format(60))
        files_num += 1
        p_id = 0
        final_output = []
        rel_num = 0
        for line in fr.readlines():
            src_data = json.loads(line.strip())

            text = src_data['text']
            spo_list = src_data['spo_list']

            spo_count = Counter()
            spo_source = dict()
            new_spo_list = []

            for spo in spo_list:
                spo2text = spo_to_text(spo)
                spo_count[spo2text] += 1
                if spo2text not in spo_source:
                    spo_source[spo2text] = spo

            for files in total_files:
                data_ = json.loads(files[p_id].strip())
                text_ = data_['text']
                assert text == text_, print(text, text_)
                spo_list = data_['spo_list']
                for spo in spo_list:
                    spo2text = spo_to_text(spo)
                    spo_count[spo2text] += 1
                    if spo2text not in spo_source:
                        spo_source[spo2text] = spo

            for k, v in spo_count.items():
                if v >= 26:
                    new_spo_list.append(spo_source[k])
            if len(new_spo_list) == 0:
                null_num += 1
            rel_num += len(new_spo_list)
            final_output.append((text, new_spo_list))
            p_id += 1
        print('rel_num', rel_num)
        print('null_num', null_num)
        print('files_num', files_num)
    with codecs.open('final_commit/final_rel_1.json', 'w', 'utf-8') as f:
        for (text, new_spo_list) in final_output:
            out_put = {}
            out_put['text'] = text
            out_put['spo_list'] = new_spo_list
            json_str = json.dumps(out_put, ensure_ascii=False)
            f.write(json_str)
            f.write('\n')


if __name__ == '__main__':

    data_preprocess('user_data/res_data/')
