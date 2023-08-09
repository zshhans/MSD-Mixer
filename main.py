import torch
import argparse
from experiments import *


def get_parser():
    parser = argparse.ArgumentParser(
        description=
        "MSD-Mixer: A Multi-Scale Decomposition MLP-Mixer for Time Series Analysis"
    )

    subparsers = parser.add_subparsers()
    ltf_subparser = subparsers.add_parser("ltf", help="Long-Term Forecasting")
    ltf_subparser.set_defaults(func=run_ltf)
    ltf_subparser.add_argument("--gpus", type=int, nargs="*", default=0)
    ltf_subparser.add_argument(
        "--dataset",
        type=str,
        nargs="*",
        default=["all"],
        help="all/ecl/etth1/etth2/ettm1/ettm2/traffic/weather/exchange")
    ltf_subparser.add_argument("--pred_len",
                               type=str,
                               nargs="*",
                               default=["all"],
                               help="all/96/192/336/720")

    stf_subparser = subparsers.add_parser("stf", help="Short-Term Forecasting")
    stf_subparser.set_defaults(func=run_stf)
    stf_subparser.add_argument(
        "--dataset",
        type=str,
        nargs="*",
        default=["all"],
        help="all/yearly/quarterly/monthly/weekly/daily/hourly")

    imp_subparser = subparsers.add_parser("imp", help="Imputation")
    imp_subparser.set_defaults(func=run_imp)
    imp_subparser.add_argument("--gpus", type=int, nargs="*", default=0)
    imp_subparser.add_argument(
        "--dataset",
        type=str,
        nargs="*",
        default=["all"],
        help="all/ecl/etth1/etth2/ettm1/ettm2/traffic/weather/exchange")
    imp_subparser.add_argument("--mask_rate",
                               type=str,
                               nargs="*",
                               default=["all"],
                               help="all/0.125/0.25/0.375/0.5")

    ad_subparser = subparsers.add_parser("ad", help="Anomaly Detection")
    ad_subparser.set_defaults(func=run_ad)
    ad_subparser.add_argument("--gpus", type=int, nargs="*", default=0)
    ad_subparser.add_argument("--dataset",
                              type=str,
                              nargs="*",
                              default=["all"],
                              help="all/smd/msl/smap/swat/psm")

    cls_subparser = subparsers.add_parser("cls", help="Classification")
    cls_subparser.set_defaults(func=run_cls)
    cls_subparser.add_argument("--gpus", type=int, nargs="*", default=0)
    cls_subparser.add_argument("--dataset",
                               type=str,
                               nargs="*",
                               default=["all"],
                               help="all/awr/af/ct/cr/fd/fm/mi/scp1/scp2/uwgl")
    return parser


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('medium')
    args = get_parser().parse_args()
    print(args)
    args.func(args)
