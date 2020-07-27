import pdb
import argparse

parser = argparse.ArgumentParser(description='Video action recogniton training')
# parser.add_argument('integers', metavar='N', type=int, nargs='+',
#                     help='an integer for the accumulator')str
# parser.add_argument('--sum', dest='accumulate', action='store_const',
#                     const=sum, default=max,
#                     help='sum the integers (default: find the max)')

parser.add_argument('--logfile_name', type=str, default="logname.log",
                    help='file name for storing the log file')
parser.add_argument('--epoch', type=int, default=10,
                    help='Number of epochs')
parser.add_argument('--gpu', type=int, default=0,
                    help='GPU ID, start from 0')
args = parser.parse_args()
# print(args.accumulate(args.integers))
# print(args.logfile_name)
print(args)
# pdb.set_trace()
# for arg in args:
# 	print(arg)