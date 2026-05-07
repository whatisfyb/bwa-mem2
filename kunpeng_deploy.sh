#!/bin/bash
# 鲲鹏ECS部署与验证脚本
# 用法: bash kunpeng_deploy.sh
# 前提: 鲲鹏ECS已配置SSH访问，git已安装

set -e

echo "=== Step 1: 克隆代码 ==="
cd ~
if [ -d "bwa-mem2-arm" ]; then
    echo "目录已存在，跳过克隆"
    cd bwa-mem2-arm
    git pull
else
    git clone <YOUR_REPO_URL> bwa-mem2-arm
    cd bwa-mem2-arm
fi
git submodule update --init --recursive

echo "=== Step 2: 编译 bwa-mem2 ARM版本 ==="
# 确认架构
echo "当前架构: $(uname -m)"
if [ "$(uname -m)" != "aarch64" ]; then
    echo "ERROR: 需要在ARM64(aarch64)机器上运行"
    exit 1
fi

# 初始化safestringlib submodule (如果还没初始化)
cd ext/safestringlib
make clean 2>/dev/null || true
cd ../..

# 编译 (使用arm64目标)
make arm64

echo "=== Step 3: 验证二进制文件 ==="
file bwa-mem2.sse41
file bwa-mem2
./bwa-mem2 version
./bwa-mem2.sse41 version

echo "=== Step 4: 小规模功能测试 ==="
mkdir -p /tmp/bwa-test
cd /tmp/bwa-test

# 创建小参考序列
python3 -c "
with open('ref.fa', 'w') as f:
    f.write('>ref\n')
    f.write('ACGT' * 500 + '\n')
"

# 创建reads
python3 -c "
with open('reads.fq', 'w') as f:
    for i in range(100):
        f.write(f'@read{i}\n')
        f.write('ACGT' * 50 + '\n')
        f.write('+\n')
        f.write('I' * 200 + '\n')
"

# 建索引
~/bwa-mem2-arm/bwa-mem2.sse41 index ref.fa

# 比对
~/bwa-mem2-arm/bwa-mem2.sse41 mem ref.fa reads.fq > arm.sam 2> arm.log

# 检查SAM输出
echo "ARM SAM行数: $(wc -l < arm.sam)"
echo "ARM NM:i:0行数: $(grep -c 'NM:i:0' arm.sam)"

cd ~/bwa-mem2-arm

echo "=== Step 5: x86正确性比对 ==="
echo "此步骤需要在x86机器上生成参考SAM，然后diff对比"
echo "步骤如下:"
echo "  1. 在x86机器上: bwa-mem2 index ref.fa && bwa-mem2 mem ref.fa reads.fq > x86.sam"
echo "  2. 将x86.sam传输到鲲鹏ECS"
echo "  3. 在鲲鹏上: diff x86.sam arm.sam"
echo "  期望结果: 无差异或仅CL行差异(CL行包含执行路径，路径不同正常)"

echo "=== Step 6: GRCh38大规模测试 ==="
echo "此步骤需要GRCh38参考基因组(3.1GB)和WGS测试数据"
echo "步骤如下:"
echo "  1. 下载GRCh38: wget <URL>/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
echo "  2. 建索引: ./bwa-mem2.sse41 index GRCh38.fa"
echo "  3. 比对: ./bwa-mem2.sse41 mem -t 64 GRCh38.fa reads.fq > grch38_arm.sam"
echo "  4. 记录时间: time ./bwa-mem2.sse41 mem -t 64 GRCh38.fa reads.fq"
echo "  5. 与BWA+jemalloc基线对比"

echo "=== Step 7: jemalloc集成测试 ==="
echo "  make USE_JEMALLOC=1 arm64"
echo "  time ./bwa-mem2.sse41 mem -t 64 GRCh38.fa reads.fq"

echo "=== 完成 ==="