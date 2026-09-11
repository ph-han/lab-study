set -e  # 에러 나면 바로 종료

TARGET_DIR="./dataset"

echo "Deleting *.png and *.csv under: $TARGET_DIR"

find "$TARGET_DIR" -type f \( -name '*.png' -o -name '*.csv' \) -print -delete

echo "Done."
