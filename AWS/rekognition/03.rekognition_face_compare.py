import os, sys
import pprint, argparse
import boto3


def main(args):
    rekognition = boto3.client('rekognition')

    if not args.source_img or not args.target_img:
        raise Exception('Missing argument(s): sourceimg or targetimg.')

    if args.s3_bucket:
        # use images from s3 bucket when specified
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
        # use images from local
        with open(args.source_img, 'rb') as f:
            source_img = f.read()
        
        with open(args.target_img, 'rb') as f:
            target_img = f.read()
        
        source_img_kwargs = {'Bytes': source_img}
        target_img_kwargs = {'Bytes': target_img}

    response = rekognition.compare_faces(
        SourceImage=source_img_kwargs,
        TargetImage=target_img_kwargs
    )

    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(response)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to try rekognition:detect_label()')
    parser.add_argument('--source-img', type=str, required=True)
    parser.add_argument('--target-img', type=str, required=True)
    parser.add_argument('--s3-bucket', type=str, default=None)

    args = parser.parse_args()

    main(args)

