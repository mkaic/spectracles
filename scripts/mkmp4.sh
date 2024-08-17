ffmpeg \
-framerate 10 \
-i spectracles/recon/images/%3d0.jpg \
-vcodec libx264 \
-pix_fmt yuv420p \
-crf 18 \
-vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
"spectracles/recon/recon.mp4" -y

# rm spectracles/recon/images/*.jpg