import os, sys
import pprint, argparse
import boto3


def main(args):
    # Define client object to connect to rekognition API
    rekognition = boto3.client('rekognition')

    # Configure argument for rekognition.Image
    if args.s3bucket and args.file_path:
        # case: specify API input using image in AWS S3
        img_kwarg = {
            'S3Object': {
                'Bucket': args.s3bucket, # path to your s3 folder
                'Name': args.file_path,   # name of the file in s3 folder, e.g. (mypic.jpg)
            }
        }

    elif args.file_path:
        # case: uploading image from local machine to rekognition API
        if os.path.isfile(args.file_path):
            # using image file
            with open(args.file_path, 'rb') as f:
                photo_bytes = f.read()

            # configure argument for rekognition.Image    
            img_kwarg = {'Bytes': photo_bytes}
        
        else:
            # error when your local image path is wrong
            raise FileExistsError('File not found in local: %s' % args.file_path)
    
    # Call Rekognition: detect_labels API
    response = rekognition.detect_labels(
        Image=img_kwarg,
        MaxLabels=10,
        MinConfidence=80
    )

    # Printing HTTP Response and its data from API call
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(response)

    return



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to try rekognition:detect_label()')
    parser.add_argument('--file-path', type=str,
        help='File path to Amazon S3 or your local file path')
    parser.add_argument('--s3bucket', type=str,
        help='If specified, the API will search `file-path` as a file path in Amazon S3 directory instead')

    args = parser.parse_args()

    main(args)