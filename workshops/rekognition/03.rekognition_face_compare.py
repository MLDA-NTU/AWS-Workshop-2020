import os, sys
import pprint, argparse
import boto3


def main(args):
    rekognition = boto3.client('rekognition')

    if not args.source_img or not args.target_img:
        raise Exception('Missing argument(s): `source-img` or `target-img`.')

    # Configure argument for rekognition.Image
    if args.s3_bucket:
        # case: specify API input using image in AWS S3
        source_img_kwargs = {
            'S3Object': {
                'Bucket': args.s3_bucket,
                'Name': args.source_img,
            }
        }
        target_img_kwargs = {
            'S3Object': {
                'Bucket': args.s3_bucket,
                'Name': args.target_img
            }
        }

    else:
        # case: using both (source and target) images from local machine
        with open(args.source_img, 'rb') as f:
            # upload source image
            source_img = f.read()
        
        with open(args.target_img, 'rb') as f:
            # upload target image
            target_img = f.read()
        
        source_img_kwargs = {'Bytes': source_img}
        target_img_kwargs = {'Bytes': target_img}

    # Call Rekognition: detect_labels API
    response = rekognition.compare_faces(
        SourceImage=source_img_kwargs,
        TargetImage=target_img_kwargs
    )

    # Printing HTTP Response and its data from API call
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(response)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to try rekognition:detect_label()')
    parser.add_argument('--source-img', type=str, required=True, 
        help='Source image path to Amazon S3 or your local file path')
    parser.add_argument('--target-img', type=str, required=True, 
        help='File path to Amazon S3 or your local file path')
    parser.add_argument('--s3-bucket', type=str, default=None, 
        help='If specified, the API will search `file-path` as a file path in Amazon S3 directory instead')

    args = parser.parse_args()

    main(args)

