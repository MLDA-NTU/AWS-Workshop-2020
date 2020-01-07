import os, sys
import pprint, argparse
import boto3


def main(args):
    rekognition = boto3.client('rekognition')

    # Configure argument for rekognition.Image
    if args.s3_bucket and args.filepath:
        # case: specify API input using image in AWS S3
        img_kwarg = {
            'S3Object': {
                'Bucket': args.s3_bucket,
                'Name': args.filepath,
            }
        }

    elif args.filepath:
        # case: uploading image from local machine to rekognition API
        if os.path.isfile(args.filepath):
            # using image file
            with open(args.filepath, 'rb') as f:
                photo_bytes = f.read()

            # configure argument for rekognition.Image    
            img_kwarg = {'Bytes': photo_bytes}

        else:
            # error when your local image path is wrong
            raise FileExistsError('File not found in local: %s' % args.filepath)

    # Call Rekognition: detect_labels API
    response = rekognition.detect_faces(
        Image=img_kwarg,
        Attributes=[args.face_attr.upper()]
    )

    # Printing HTTP Response and its data from API call
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(response)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to try rekognition:detect_label()')
    parser.add_argument('--file-path', type=str, help='File path to Amazon S3 or your local file path')
    parser.add_argument('--s3-bucket', type=str, help='If specified, the API will search `file-path` as a file path in Amazon S3 directory instead')
    parser.add_argument('--face-attr', type=str, default='DEFAULT', help='Face attributes to be specified as input to Rekognition API.')

    args = parser.parse_args()

    main(args)