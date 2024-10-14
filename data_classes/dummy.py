import json

if __name__ == '__main__':
    zsre_impl_path = '../data/zsre_dummy/impl_{}.json'
    zsre_yn_path = '../data/zsre_dummy/zsre_yn_{}.txt'
    zsre_path = '../data/zsre_dummy/structured_zeroshot-{}-new_annotated_final.jsonl'

    # impl
    # for split in ["train", "dev", "test"]:
    #     print(zsre_impl_path.format(split))
    #     with open(zsre_impl_path.format(split)) as f:
    #         data = json.load(f)
    #         for impl in data:
    #             for q in impl:
    #                 q[1] = 'dummy'
    #     with open(zsre_impl_path.format(split), 'w') as f:
    #         json.dump(data, f)


    for split in ["train", "dev", "test"]:
        print(zsre_path.format(split))
        with open(zsre_path.format(split)) as f:
            data = [json.loads(line) for line in f]
            for item in data:
                for output in item['output']:
                    output['answer'] = 'dummy'
                    output['provenance'] = []
        with open(zsre_path.format(split), 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')

# todo 
    # ck that provencance = [] is okay
    # create separate config for this expeirment for now just changed the og for qa to the new dummy path
    # restore old yn (?) files to their og form since i accidentally edited