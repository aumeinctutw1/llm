import requests
import argparse

def download_file(url, output_file):
    response = requests.get(url)
    with open(output_file, 'wb') as f:
        f.write(response.content)

def get_file(url):
    response = requests.get(url)
    return response.content

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', help='URL to download')
    parser.add_argument('--output', help='Output file')
    args = parser.parse_args()

    if args.url and args.output:
        download_file(args.url, args.output)
    else:
        print('Please provide both --url and --output')