import argparse

def main():
    # Argument parser 생성
    parser = argparse.ArgumentParser(description="Simple CLI calculator using argparse")
    
    # 필요한 인자 추가
    parser.add_argument('x', type=float, help='First number')
    parser.add_argument('y', type=float, help='Second number')
    parser.add_argument('operation', choices=['add', 'sub', 'mul', 'div'], help='Operation to perform')
    
    # 파싱
    args = parser.parse_args()

    # 연산 수행
    result = None
    if args.operation == 'add':
        result = args.x + args.y
    elif args.operation == 'sub':
        result = args.x - args.y
    elif args.operation == 'mul':
        result = args.x * args.y
    elif args.operation == 'div':
        if args.y != 0:
            result = args.x / args.y
        else:
            result = 'Error: Division by zero!'
    
    # 결과 출력
    print(f'Result: {result}')

if __name__ == "__main__":
    main()
