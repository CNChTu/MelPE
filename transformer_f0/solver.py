import numpy as np
import torch
from .saver.saver import Saver
from .saver import utils
from torch import autocast
from torch.cuda.amp import GradScaler


def test(args, model, loader_test, saver):
    print(' [*] testing...')
    model.eval()

    # losses
    test_loss = 0.

    # intialization
    num_batches = len(loader_test)
    rtf_all = []

    # run
    with torch.no_grad():
        for bidx, data in enumerate(loader_test):
            fn = data['name'][0]
            print('--------')
            print('{}/{} - {}'.format(bidx, num_batches, fn))

            # unpack data
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(args.device)
            print('>>', data['name'][0])

            # forward

            f0 = model(mel=data['mel'], infer=True)

            # loss
            for i in range(args.train.batch_size):
                loss = model(mel=data['mel'], infer=False, gt_f0=data['f0'])
                test_loss += loss.item()

            # log mel
            saver.log_spec(data['name'][0], data['mel'], data['mel'])

            saver.log_f0(data['name'][0], f0, data['f0'])

    # report
    test_loss /= args.train.batch_size
    test_loss /= num_batches

    # check
    print(' [test_loss] test_loss:', test_loss)
    print(' Real Time Factor', np.mean(rtf_all))
    return test_loss


def train(args, initial_global_step, model, optimizer, scheduler, loader_train, loader_test):
    # saver
    saver = Saver(args, initial_global_step=initial_global_step)

    # model size
    params_count = utils.get_network_paras_amount({'model': model})
    saver.log_info('--- model size ---')
    saver.log_info(params_count)

    # run
    num_batches = len(loader_train)
    model.train()
    saver.log_info('======= start training =======')
    scaler = GradScaler()
    if args.train.amp_dtype == 'fp32':
        dtype = torch.float32
    elif args.train.amp_dtype == 'fp16':
        dtype = torch.float16
    elif args.train.amp_dtype == 'bf16':
        dtype = torch.bfloat16
    else:
        raise ValueError(' [x] Unknown amp_dtype: ' + args.train.amp_dtype)
    for epoch in range(args.train.epochs):
        for batch_idx, data in enumerate(loader_train):
            saver.global_step_increment()
            optimizer.zero_grad()

            # unpack data
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(args.device)

            # forward
            if dtype == torch.float32:
                loss = model(mel=data['mel'], infer=False, gt_f0=data['f0'])
            else:
                with autocast(device_type=args.device, dtype=dtype):
                    loss = model(mel=data['mel'], infer=False, gt_f0=data['f0'])

            # handle nan loss
            if torch.isnan(loss):
                raise ValueError(' [x] nan loss ')
            else:
                # backpropagate
                if dtype == torch.float32:
                    loss.backward()
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                scheduler.step()

            # log loss
            if saver.global_step % args.train.interval_log == 0:
                current_lr = optimizer.param_groups[0]['lr']
                saver.log_info(
                    'epoch: {} | {:3d}/{:3d} | {} | batch/s: {:.2f} | lr: {:.6} | loss: {:.3f} | time: {} | step: {}'.format(
                        epoch,
                        batch_idx,
                        num_batches,
                        args.env.expdir,
                        args.train.interval_log / saver.get_interval_time(),
                        current_lr,
                        loss.item(),
                        saver.get_total_time(),
                        saver.global_step
                    )
                )

                saver.log_value({
                    'train/loss': loss.item()
                })

                saver.log_value({
                    'train/lr': current_lr
                })

            # validation
            if saver.global_step % args.train.interval_val == 0:
                optimizer_save = optimizer if args.train.save_opt else None

                # save latest
                saver.save_model(model, optimizer_save, postfix=f'{saver.global_step}')
                last_val_step = saver.global_step - args.train.interval_val
                if last_val_step % args.train.interval_force_save != 0:
                    saver.delete_model(postfix=f'{last_val_step}')

                # run testing set
                test_loss = test(args, model, loader_test, saver)

                # log loss
                saver.log_info(
                    ' --- <validation> --- \nloss: {:.3f}. '.format(
                        test_loss,
                    )
                )

                saver.log_value({
                    'validation/loss': test_loss
                })

                model.train()
