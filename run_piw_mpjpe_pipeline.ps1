$ErrorActionPreference = "Continue"
$py = "C:\Users\toand\AppData\Local\Programs\Python\Python314\python.exe"
Set-Location "C:\Users\toand\.openclaw\workspace\Decompose_WiFi_Sensing_HPE"
$log = "outputs\piw_mpjpe_pipeline_full.log"
"[start] $(Get-Date -Format o)" | Out-File $log -Encoding utf8

# Step 1: retrain anytime model, dump per-frame MPJPE-by-rank + stratified A(R) by person count
& $py experiments\exp31_piw_anytime_pra.py --epochs 80 2>&1 | Out-File $log -Append -Encoding utf8
"[exp31 done] $(Get-Date -Format o)" | Out-File $log -Append -Encoding utf8

# Step 2: MPJPE contention sweep, single-CP (cp-count=1), placeholder rank-4 anchor 1300us
foreach ($d in 15,20,30,40,60) {
  "[exp33 D=$d] $(Get-Date -Format o)" | Out-File $log -Append -Encoding utf8
  & $py experiments\exp33_pra_mpjpe_contention.py `
      --rank-eval outputs\piw_anytime_pra_rank_eval.npz `
      --cp-count 1 --cp-us-at-rank4 1300 --deadline-ms $d `
      --csv "results\piw_mpjpe_D$d.csv" `
      --plot "PAPER\figures\fig_piw_mpjpe_D$d.png" 2>&1 | Out-File $log -Append -Encoding utf8
}
"[all done] $(Get-Date -Format o)" | Out-File $log -Append -Encoding utf8
