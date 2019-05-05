from __future__ import print_function, absolute_import, division
import argparse
import sys

description = """Make a submission to the challenge. Examples:
{executable} submit FILE
{executable} submit FILE --submit_comment 'Tried another approach ...'
""".format(executable=sys.argv[0])

try:
    import requests
except:
    print('Please install "requests" package first (pip install requests)')

server = 'http://www.compai-msu.info/'
auth_options = dict(
    challenge_name='spring2019_ilimdb_sentiment_assignment2',
    user_email= 'danil.kononyhin@mail.ru',
    user_token='c3d1470b53665a4a97c517af82bc697f',
)
proxies = None

parser = argparse.ArgumentParser(usage=description)
subparsers = parser.add_subparsers(title='valid operations', dest='COMMAND')
subparsers.required = True
submit_parser = subparsers.add_parser('submit')
submit_parser.add_argument('submit_filename', metavar='FILE', type=str, help='File for submission')
submit_parser.add_argument('--submit_comment', metavar='COMMENT', type=str, default='',
                           help='Optional comment describing this submission')


def submit(filename, user_comment):
    data = dict(user_comment=user_comment, **auth_options)
    files = dict(file=(filename, open(filename, 'rb')))
    response = requests.post(server + '/api/submit', data=data, files=files, proxies=proxies)
    print(response.text)


if __name__ == '__main__':
    args = parser.parse_args()
    submit(filename=args.submit_filename, user_comment=args.submit_comment)
