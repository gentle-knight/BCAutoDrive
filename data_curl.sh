# ========== 只修改这4行 ==========
TOKEN="0a9840b24f6f49c1ee39ed7bd58f7f76bba6e068"        # 替换成你的API token
REPO_ID="2e47785f-2cab-43a1-bce7-1608bea9edb0"       # 替换成你的资料库ID
LOCAL_DIR="/home/huangfukk/workspace/MAGAIL4AutoDrive/legacy_magail/data/exp_converted"   # 替换成你要上传的本地文件夹
REMOTE_DIR="/无限场景生成/专家数据集"     # 替换成BOX里的目标路径
# ================================

find "$LOCAL_DIR" -type f | while read file; do
    rel_path="${file#$LOCAL_DIR/}"
    curl -s -X POST \
      -H "Authorization: Token $TOKEN" \
      -F "file=@$file" \
      "https://box.nju.edu.cn/api2/repos/$REPO_ID/upload-file/?p=$REMOTE_DIR/$rel_path"
    echo "✅ 已上传: $rel_path"
done
