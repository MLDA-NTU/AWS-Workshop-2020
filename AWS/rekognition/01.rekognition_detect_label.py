import os, sys
import pprint, argparse
import boto3


def main(args):
    rekognition = boto3.client('rekognition')

    if args.s3bucket and args.filepath:
        # configure argument for rekognition.Image
        img_kwarg = {
            'S3Object': {
                'Bucket': args.s3bucket,
                'Name': args.filepath,
            }
        }

    elif args.filepath:
        if os.path.isfile(args.filepath):
            # using image file
            with open(args.filepath, 'rb') as f:
                photo_bytes = f.read()

            # configure argument for rekognition.Image    
            img_kwarg = {'Bytes': photo_bytes}
        
        else:
            raise FileExistsError('File not found in local: %s' % args.filepath)
    
    response = rekognition.detect_labels(
        Image=img_kwarg,
        MaxLabels=10,
        MinConfidence=80
    )

    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(response)

    return



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to try rekognition:detect_label()')
    parser.add_argument('--filepath', type=str)
    parser.add_argument('--s3bucket', type=str)

    args = parser.parse_args()

    main(args)