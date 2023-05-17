def main():
    while True:
        results = []
        while True:
            line = input()
            if line != '':
                results.append([float(r) for r in line.split('\t')])
            else:
                break

        for col in zip(*results):
            print(max(col), end='\t')
        print('\n')

        rs = []
        for i, row in enumerate(results):
            rs.append((sum(row[1:4]), sum(row), i))
        max_id = max(rs)[2]

        print(*(results[max_id]), sep='\t')


if __name__ == "__main__":
    main()
