# import pandas as pd
import os
import json
import requests
import csv
from argparse import ArgumentParser
'''
argument template
                           target dir                                  cred file                                                    caseid csv
python downlod.py /home/aindra/annot_cases_cpap/cpap /home/aindra/annot_cases_cpap/clustrtrain-3821fb432e2f.json /home/aindra/annot_cases_cpap/caseid_file.csv

'''

def download_pyr_dir(pyr_dir_url, tgt_dir, cred_file):
    auth_postfix = 'auth activate-service-account --key-file={}'.format(cred_file)
    auth_command = 'gcloud' + ' {}'.format(auth_postfix)
    do_authentication = os.system(auth_command)
    if do_authentication == 0:
        pyr_dir_url = pyr_dir_url.replace('https://storage.googleapis.com/', '')
        download(pyr_dir_url, tgt_dir)
    else:
        print("error")


def download(pyr_dir_url, tgt_dir):
    cmd_path = "gsutil -m rsync -d -r gs://{} {}".format(pyr_dir_url, tgt_dir)
    download_command = os.system(cmd_path)
    if download_command == 0:
        print("Successfully downloaded for slide {}".format(tgt_dir))
    else:
        print("Download failed for slide {}".format(tgt_dir))


if __name__ == "__main__":

    if __name__ == "__main__":
        parser = ArgumentParser(description="Download data from clustr between given dates")

        parser.add_argument(dest="batch_dir", help="Batch directory", metavar="$BATCH_DIR$")
        parser.add_argument(dest="cred_file", help="Credential file of google storage bucket", metavar="$CRED$")
        parser.add_argument(dest="caseid_file", help="File with list of case id", metavar="$CASEID_LIST$")

        args = parser.parse_args()
        # cases_info_query = dict()
        # cases_info_query['startDate'] = args.start_date
        # cases_info_query['endDate'] = args.end_date

        print(args.batch_dir)
        with open(args.cred_file, 'r') as f:
            credentials = json.load(f)

        file = open(args.caseid_file)
        csvreader = csv.reader(file)

        # wb = xlrd.open_workbook(args.caseid_file)
        # sheet = wb.sheet_by_index(0)
        # sheet.cell_value(0, 0)
        print("reading csv")
        for row in csvreader:
            case_id = row[0].strip()
            # print(case_id)
            bucket_name = credentials['bucket_name']

            # response = requests.get(
            #     'https://{}/_ah/clustrApi/cases/caseid-data?case_id={}'.format(bucket_name, case_id))
            print("calling api")
            response = requests.get(
                'https://clustrtrain.appspot.com/_ah/clustrApi/cases/annotdetail-V2?case_id={}'.format(case_id))
            print(response.status_code)
            # exit(0)
            if response.status_code == 200:

                case_data = response.json()
                case_slides_info = case_data['caseSlides']
                for case_slide in case_slides_info:
                    pyr_url = case_slide['slideImageURNPath']
                    pyr_url = pyr_url.rstrip('/')
                    # print('new url', pyr_url, case_id)
                    # if 'VISIO"	NX_DATA' in pyr_url:
                    pyr_url_items = pyr_url.split('/')
                    patient_id, slide_id = pyr_url_items[-3], pyr_url_items[-2]
                    case_name = case_slide['caseSlideID']
                    # print((case_name), (patient_id), (slide_id))
                    tgt_dir = os.path.join(args.batch_dir, patient_id, slide_id, "pyramid")

                    if not os.path.exists(tgt_dir):
                        os.makedirs(tgt_dir)
                    # print(case_slide.keys())
                    # print(case_slide['slideRoiAnnotation'])
                    if 'slideRoiAnnotation' in case_slide.keys() and case_slide['slideRoiAnnotation'] != "":
                        # print("{}-has slide roi annotations".format(case_slide))
                        annot_data = case_slide['slideRoiAnnotation']
                        annot_file_path = os.path.join(args.batch_dir, patient_id, slide_id, 'annot.json')
                        fmt_annot_data = json.loads(annot_data)
                        with open(annot_file_path, 'w') as f:
                            json.dump(fmt_annot_data, f, indent=4)

                    case_info = dict()
                    case_info['caseSlideId'] = case_name
                    # case_info['caseManager'] = case_originator
                    # print(case_data)
                    with open(os.path.join(args.batch_dir, patient_id, slide_id, 'info.json'), 'w') as f:
                        json.dump(case_info, f, indent=4)
                    download_pyr_dir(pyr_url, tgt_dir, args.cred_file)