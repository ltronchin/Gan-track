import requests

def download_file_from_google_drive(fil_id, dest):
    print('Start')
    URL = "https://drive.google.com/uc?export=download" # "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : fil_id }, stream = True)
    token = get_confirm_token(response=response)

    if token:
        params = { 'id' : fil_id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response=response, dest=dest)

    print('Done.')

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, dest):
    CHUNK_SIZE =  32768

    with open(dest, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


# importing the requests module

def download_zip_file_from_url():

    print('Start')
    url = 'https://drive.google.com/drive/u/0/folders/1Hoc7Ei-JXQESKRKQa12971FaBORiUGOJ'

    # Downloading the file by sending the request to the URL
    req = requests.get(url)

    # Split URL to get the file name
    filename = url.split('/')[-1]

    # Writing the file to the local file system
    with open(filename, 'wb') as output_file:
        output_file.write(req.content)

    print('Done.')

if __name__ == "__main__":
    # https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive
    file_id = '1oB11SHNUaGL-KgXVYdy94DCI26cg8WGQ' # '<file_id> from sharable link in google drive
    destination = '/home/lorenzo/data_m2/data/raw/claro/test.tar' # destination file on your disk
    download_file_from_google_drive(fil_id=file_id, dest=destination)