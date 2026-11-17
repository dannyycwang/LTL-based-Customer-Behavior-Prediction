def seed_everything(seed: int=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_list(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    return json.loads(x)

def make_trace(row, pack, tail_len=10, product_mode='semantic'):
    actions = parse_list(row['action_sequence'] if 'action_sequence' in row else row['behavior_sequence'])
    dwells = parse_list(row['dwell_bucket_sequence']) if 'dwell_bucket_sequence' in row else []
    atoms = parse_list(row['atom_sequence']) if 'atom_sequence' in row else []
    n = len(actions)
    if pack in ('BD', 'BDP'):
        n = min(n, len(dwells))
    if pack in ('BP', 'BDP') and product_mode == 'semantic':
        n = min(n, len(atoms))
    if pack in ('BP', 'BDP') and product_mode == 'sku_hash':
        pseq = parse_list(row['product_sku_hash_sequence'])
        n = min(n, len(pseq))
    if pack in ('BP', 'BDP') and product_mode == 'category_hash':
        cseq = parse_list(row['category_hash_sequence'])
        n = min(n, len(cseq))
    start = max(0, n - tail_len)
    trace = []
    for i in range(start, n):
        step = []
        step.append(str(actions[i]))
        if pack in ('BD', 'BDP'):
            step.append(str(dwells[i]))
        if pack in ('BP', 'BDP'):
            if product_mode == 'semantic':
                for tok in atoms[i]:
                    tok = str(tok)
                    if tok not in ACTION_TOKENS and tok not in DWELL_TOKENS:
                        step.append(tok)
            elif product_mode == 'sku_hash':
                pseq = parse_list(row['product_sku_hash_sequence'])
                step.append(f'p_{pseq[i]}')
            elif product_mode == 'category_hash':
                cseq = parse_list(row['category_hash_sequence'])
                step.append(f'c_{cseq[i]}')
            else:
                raise ValueError(f'Unknown product_mode: {product_mode}')
        step = sorted(set((t for t in step if t not in ['', 'nan', 'None'])))
        trace.append(step)
    return trace

def build_pack_json(df, pack, out_json):
    pos_df = df[df[LABEL_COL] == 1].sample(n=SAMPLE_POS, random_state=FORMULA_SEED)
    neg_df = df[df[LABEL_COL] == 0].sample(n=SAMPLE_NEG, random_state=FORMULA_SEED)
    traces_pos = [make_trace(row, pack, TRACE_TAIL_LEN, PRODUCT_MODE) for (_, row) in pos_df.iterrows()]
    traces_neg = [make_trace(row, pack, TRACE_TAIL_LEN, PRODUCT_MODE) for (_, row) in neg_df.iterrows()]
    traces_pos = [t for t in traces_pos if len(t) > 0]
    traces_neg = [t for t in traces_neg if len(t) > 0]
    vocab = sorted({atom for trace in traces_pos + traces_neg for step in trace for atom in step})
    data = {'vocab': vocab, 'traces_pos': traces_pos, 'traces_neg': traces_neg, 'ltlftree': ''}
    with open(out_json, 'w') as f:
        json.dump(data, f)
    print(f'\n[{pack}] saved:', out_json)
    print(f'  pos={len(traces_pos)}, neg={len(traces_neg)}, vocab={len(vocab)}')
    print(f'  vocab={vocab}')
    return data

def train_tltlf_model(train_file, save_model, netlen=10, epoch_num=100, lr=0.01, batch_size=100, a1=0.1, a2=0.1, a3=0.1):
    seed_everything(FORMULA_SEED)
    (train_dataset, word2idx, vocab, ltlf_tree) = get_data(str(train_file))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    model = Net(netlen, len(vocab))
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    best_loss = float('inf')
    start_time = time.time()
    for epoch in range(epoch_num):
        model.train()
        epoch_loss = 0.0
        for (data_x, data_y, predict_mask) in train_loader:
            data_x = data_x.to(DEVICE)
            data_y = data_y.to(DEVICE)
            predict_mask = predict_mask.to(DEVICE)
            optimizer.zero_grad()
            prediction = model(data_x)
            main_loss = torch.sum((prediction[:, :, :1] * predict_mask - data_y) ** 2)
            reg_loss = torch.tensor(0.0, device=DEVICE)
            temp_a = model.chose_right_.transpose(0, 1)
            temp_b = model.op_chose_weight.transpose(0, 1)
            for i in range(1, netlen):
                reg_loss = reg_loss + a1 * torch.pow(torch.sum(temp_a[i]) + torch.sum(temp_b[i - 1]) - 1, 2)
            reg_loss = reg_loss + torch.sum(a2 * torch.relu(model.op_chose_weight[0][:netlen - 1] - model.op_chose_weight[0][1:]))
            for i in range(netlen - 2):
                for j in range(i + 2, netlen):
                    reg_loss = reg_loss + torch.sum(torch.relu(model.chose_right_[i][j] + model.chose_right_[i + 1:j, j + 1:netlen] - 1) * a3)
            loss = main_loss + reg_loss
            loss.backward()
            optimizer.step()
            epoch_loss += float(main_loss.detach().cpu())
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), str(save_model))
        print(f'epoch={epoch:03d} | main_loss={epoch_loss:.4f} | best={best_loss:.4f} | time={time.time() - start_time:.1f}s')
        if best_loss < 1.0:
            print('Early stop: best_loss < 1.0')
            break
    return {'best_loss': best_loss, 'train_time_sec': time.time() - start_time, 'vocab_size': len(vocab), 'n_train': len(train_dataset)}

def formula_to_str(f):
    if isinstance(f, str):
        return f
    op = f[0]
    if op in ['!', 'not']:
        return f'!({formula_to_str(f[1])})'
    if op == 'X':
        return f'X({formula_to_str(f[1])})'
    if op in ['&&', 'and', '&']:
        return f'({formula_to_str(f[1])} && {formula_to_str(f[2])})'
    if op == 'U':
        return f'({formula_to_str(f[1])} U {formula_to_str(f[2])})'
    return str(f)

def evaluate_formula_on_json(formula_tree, json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    ltlf = LTLf(data['vocab'], formula_tree)
    (pos_mark, neg_mark) = ltlf.evaluate(data['traces_pos'], data['traces_neg'])
    pos_support = float(np.mean(pos_mark)) if len(pos_mark) else 0.0
    neg_support = float(np.mean(neg_mark)) if len(neg_mark) else 0.0
    support_gap = pos_support - neg_support
    balanced_accuracy = 0.5 * (pos_support + (1.0 - neg_support))
    signature = bytes([int(x) for x in pos_mark + neg_mark])
    return {'pos_support': pos_support, 'neg_support': neg_support, 'support_gap': support_gap, 'balanced_accuracy': balanced_accuracy, 'signature': signature}