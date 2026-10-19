"""Independent stdlib raw audit of the reversed whole-policy two-process pair."""

import argparse
import json
import statistics
from pathlib import Path, PurePosixPath

import running_o_audit as core

CELLS = [('control', 'whole'), ('resident', 'whole')]
IDENTITIES = (
    'input_identity', 'post_input_identity', 'mask_identity', 'post_mask_identity',
    'frame_query_identity', 'post_frame_query_identity', 'selected_records', 'scripts',
    'extension_sha256', 'torch_version', 'torch_npu_version', 'device_name',
    'trace_contract', 'installed_catalog', 'schemas', 'legacy_contract',
    'policy_provenance', 'opp_artifacts', 'python_startup_opp',
    'effective_opp_before_native_load', 'scope', 'synthetic_inputs', 'endpoint',
    'queue_span_semantics', 'canonical_digest',
)


def only(seen, name, audit):
    found = [path for path in seen if path.name == name]
    audit.same(len(found), 1, 'one manifested ' + name)
    return found[0]


def manifest(base, audit):
    seen = set()
    for item in core.load(base / 'manifest.json'):
        path = core.owned(base, item['path'])
        audit.require(path not in seen, 'unique manifested path')
        audit.same(core.sha(path), item['sha256'], 'manifest full byte hash ' + item['path'])
        seen.add(path)
    return seen


def native_identity(value):
    return {key: ([{k: v for k, v in row.items() if k != 'address'} for row in item]
                  if key == 'sink_symbols' else item) for key, item in value.items()}


def cli(command, reference, audit):
    audit.same(command[:2], reference[:2], 'same Python and frozen consumer path')
    expected = {option: reference[reference.index(option) + 1] for option in
                ('--root', '--opp-root', '--trace', '--selection', '--mode', '--trial',
                 '--repeats', '--warmups', '--profile-repeats', '--seed', '--timeout')}
    audit.same(set(command[2::2]), set(expected) | {'--output-dir'}, 'complete CLI option set')
    audit.same(len(command), 2 + 2 * (len(expected) + 1), 'complete CLI argument count')
    for option, value in expected.items():
        audit.same(command.count(option), 1, 'one exact CLI option ' + option)
        audit.same(command[command.index(option) + 1], value, 'frozen CLI option ' + option)
    for option, value in {'--mode':'whole', '--trial':'0', '--repeats':'10', '--warmups':'5',
                          '--profile-repeats':'1', '--seed':'1703'}.items():
        audit.same(expected[option], value, 'fixed pair CLI option ' + option)


def qualify(report, reference, audit):
    for key in IDENTITIES:
        audit.same(report[key], reference[key], 'same-variant exact reference identity ' + key)
    expected = dict(state='completed', stage='completed', returncode=0, mode='whole', policy='whole',
                    trial=0, repeats=10, occurrences=16, profile_repeats=1,
                    warmup_replays=5, warmup_count=5, performance_measured=True,
                    npu_work_executed=True, independent_graph_pool_per_bucket=True,
                    separate_qualification=True, timed_samples_individually_qualified=False,
                    frame_queries_match_base_prefixes=True, full_model_qualified=False,
                    full_trace_measured=False, end_to_end_gain_established=False,
                    core_limit=dict(cube_core_num=20, vector_core_num=40))
    for key, value in expected.items():
        audit.same(report[key], value, 'completed bounded probe ' + key)
    audit.same(native_identity(report['final_native_provenance']),
               native_identity(reference['final_native_provenance']),
               'native owners and library bytes unchanged; process addresses excluded')
    audit.same([{k:v for k,v in row.items() if k!='address'} for row in report['symbols']],
               [{k:v for k,v in row.items() if k!='address'} for row in reference['symbols']],
               'initial native symbol ownership unchanged')
    core.input_storage(report, audit)
    records = {row['physical_tokens']:row for row in report['selected_records']}
    audit.same(sorted(records), core.CAPS, 'original two capacities')
    audit.same([records[cap]['step'] for cap in core.CAPS], [0,29], 'original trace steps')
    audit.same([records[cap]['batch'] for cap in core.CAPS], [98,128], 'original real batches')
    audit.same(len(report['metadata_preflight']), 2, 'two eager native plan preflights')
    plans = {}
    for item, original in zip(report['metadata_preflight'], reference['metadata_preflight'], strict=True):
        cap = item['capacity']
        audit.same(item['record'], records[cap], 'preflight current original record')
        audit.same(item, original, 'full eager metadata equal same-variant reference')
        plans[cap] = core.plan_check(item, records[cap], 'whole', audit)
    rows_count, max_error, plan_count = 0, 0.0, 2
    outputs = []
    for phase in ('pre_checks', 'post_checks'):
        audit.same(len(report[phase]), 2, 'two separate native qualification records')
        for item, original in zip(report[phase], reference[phase], strict=True):
            cap, record = item['capacity'], records[item['capacity']]
            audit.same([item['step'],item['private_occurrences']], [record['step'],16], 'qualification identity')
            for flag in ('finite','all_outputs_finite','exact_zero_padding','stable_addresses','exact_descriptors_read_only'):
                audit.require(item[flag] is True, 'native qualification ' + flag)
            audit.same([item['legacy_update_calls_delta'],item['legacy_updated_tasks_delta']], [0,0], 'no substituted legacy update')
            audit.same(core.plan_check(item['plan'], record, 'whole', audit), plans[cap], 'actual replay equals eager plan')
            audit.same(item['plan'], original['plan'], 'full replay plan equals reference')
            plan_count += 1
            rows = item['fp32_sampled_rows']
            wanted = {(occ,row) for occ in (0,7,15) for row in (0,record['batch']//2,record['batch']-1)}
            audit.same({(row['occurrence'],row['row']) for row in rows}, wanted, 'nine FP32 original sampled identities')
            audit.same(len(rows), 9, 'nine independent FP32 rows')
            for row in rows:
                length = record['kv_lengths'][row['row']]
                pages = record['block_table'][row['row']][:(length+127)//128]
                audit.same([row['kv_length'],row['original_page_ids'],row['local_page_ids']],
                           [length,pages,list(range(len(pages)))], 'oracle true ordered KV pages')
                audit.require(core.finite(row['max_abs_error']), 'finite FP32 error')
                max_error = max(max_error,row['max_abs_error'])
                rows_count += 1
            audit.same(rows, original['fp32_sampled_rows'], 'FP32 diagnostics stable against same-variant reference')
            audit.same(len(item['output_hashes']), 16, 'all sixteen private output hashes')
            for tensor in item['output_hashes']:
                core.tensor(tensor,[cap,12,256],audit)
            audit.same(item['output_hashes'], original['output_hashes'], 'all output bytes equal same-variant reference')
            outputs.append(dict(capacity=cap,phase=phase,all_16_output_hashes_match_reference=True))
    audit.same(report['warmups'],[dict(capacity=cap,replays=5,outside_timing=True) for cap in core.CAPS], 'all warmups outside timing')
    samples = report['samples']
    audit.same(len(samples),20,'twenty unfiltered raw timings')
    audit.same([(row['repeat'],row['capacity']) for row in samples],
               [(repeat,cap) for repeat in range(10) for cap in (core.CAPS if repeat%2==0 else core.CAPS[::-1])],
               'frozen alternating record order')
    for row in samples:
        record = records[row['capacity']]
        audit.same([row['trial'],row['mode'],row['policy'],row['step'],row['batch']],
                   [0,'whole','whole',record['step'],record['batch']],'formal sample original identity')
        audit.require(all(core.finite(row[name]) for name in core.METRICS) and row['wall_us']>0,'finite formal metrics')
        audit.require(row['preparation_enqueue_us']+row['submit_join_cpu_us']<=row['wall_us']+1e-6,'wall endpoint includes same-clock subcosts')
        audit.same([row['legacy_update_calls_delta'],row['legacy_updated_tasks_delta']],[0,0],'formal sink producer retained')
    timing = []
    for cap in core.CAPS:
        selected = [row for row in samples if row['capacity']==cap]
        audit.same(len(selected),10,'ten unfiltered timings per record')
        metric = {name:statistics.mean(row[name] for row in selected) for name in core.METRICS}
        wall = [row['wall_us'] for row in selected]
        metric.update(wall_median_us=statistics.median(wall),wall_min_us=min(wall),wall_max_us=max(wall),
                      wall_range_us=max(wall)-min(wall),maximum_sample=max(selected,key=lambda row:row['wall_us']))
        timing.append(dict(capacity=cap,samples=10,samples_removed=0,**metric))
    return dict(timings=20,fp32_rows=rows_count,complete_plan_checks=plan_count,
                maximum_fp32_absolute_error=max_error,timing=timing,output_identity=outputs)


def audit_pair(bundle,reference,audit):
    ref_audit = core.audit_bundle(reference,core.Audit())
    audit.require(ref_audit['qualified'] is True,'raw reference independently requalified')
    reference = reference.resolve(strict=True)
    ref_seen = manifest(reference,audit)
    ref_suite = core.load(only(ref_seen,'running-o-screen.json',audit))
    ref_runs = {(row['variant'],row['policy']):row for row in ref_suite['runs']}
    ref_reports = {pair:core.load(reference/row['report_path']) for pair,row in ref_runs.items()}
    base = bundle.resolve(strict=True)
    collection = core.load(base/'collection.json')
    audit.require(collection['partial'] is False and collection['controller_interrupted'] is False,'complete uninterrupted pair collection')
    seen = manifest(base,audit)
    suite_path = only(seen,'running-o-confirm.json',audit)
    suite = core.load(suite_path)
    audit.same(collection['suite_sha256'],core.sha(suite_path),'collector binds exact pair suite bytes')
    audit.same([suite['state'],suite['returncode']],['completed',0],'completed confirmation suite')
    audit.require(suite['builds_reused_from_v3'] is True and suite['consumer_probe_unchanged'] is True,'same v3 builds and frozen probe')
    audit.require(suite['performance_measured'] is True,'formal confirmation timings exist')
    for flag in ('full_trace_measured','full_model_qualified','end_to_end_gain_established'):
        audit.require(suite[flag] is False,'scope excludes '+flag)
    audit.same(suite['design'],dict(cells=[list(pair) for pair in CELLS],repeats=10,warmups=5,
               profile_repeats=1,seed=1703,trial=0,capacities=core.CAPS,occurrences=16),'actual predeclared two-cell design')
    deployed = core.load(only(seen,'deployed-tools.json',audit))
    audit.same(deployed,core.load(only(ref_seen,'deployed-tools.json',audit)),'exact full v3 frozen tools manifest')
    audit.same(deployed['aclgraph_fia_sink_consumer_probe.py'],core.PROBE,'frozen measured consumer probe')
    for name in ('aclgraph_fia_sink_consumer_probe.py','aclgraph_fia_sink_stage7_suite.py'):
        audit.same(core.sha(only(seen,name,audit)),deployed[name],'raw runtime script full byte hash '+name)
    driver = base/'deployed-tools/running_o_pair_driver.py'
    driver_manifest = base/'deployed-tools/manifest.json'
    audit.require(driver in seen and driver_manifest in seen,'raw driver and deployment manifest included')
    audit.same(core.load(driver_manifest),{driver.name:core.sha(driver)},'exact deployed driver hash')
    builds = {}
    for variant in ('control','resident'):
        path = reference/'builds'/variant/'running-o-build.json'
        audit.require(path in ref_seen,'reference build report manifested')
        build = core.load(path)
        audit.same(suite['builds'][variant]['report_sha256'],core.sha(path),'exact reused successful v3 build report')
        audit.same(suite['builds'][variant]['artifacts'],build['artifacts'],'same exact v3 binary artifact inventory')
        builds[variant] = dict(report_sha256=core.sha(path),artifacts=build['artifacts'],restricted_tiling_keys=core.TILING_KEYS)
    audit.same(len(suite['runs']),2,'two fresh serial whole-policy processes')
    summaries,profiles,parent_pids,child_pids = [],[],set(),set()
    previous = suite['started_unix']
    for index,(entry,pair) in enumerate(zip(suite['runs'],CELLS,strict=True)):
        variant,policy = pair
        audit.same([entry['variant'],entry['policy'],entry['process_order_index']],[variant,policy,index],'reverse whole-policy order')
        audit.require(previous<=entry['launched_unix']<entry['finished_unix']<=suite['finished_unix'],'serial nonoverlapping process intervals')
        previous = entry['finished_unix']
        audit.same(entry['returncode'],0,'successful child process')
        path = core.owned(base,entry['report_path'])
        audit.require(path in seen,'manifested process report')
        audit.same(core.sha(path),entry['report_sha256'],'suite binds process exact raw bytes')
        report = core.load(path)
        cli(entry['command'],ref_runs[pair]['command'],audit)
        audit.same(PurePosixPath(entry['command'][entry['command'].index('--output-dir')+1]).name,path.parent.name,'CLI output agrees with suite-bound report')
        for value,bag in ((entry['pid'],parent_pids),(report['child_pid'],child_pids)):
            audit.require(type(value) is int and value>0 and value not in bag,'unique positive process PID')
            bag.add(value)
        result = qualify(report,ref_reports[pair],audit)
        result.update(variant=variant,policy=policy)
        summaries.append(result)
        audit.same(len(report['profiles']),2,'two independent diagnostic profile replays')
        audit.same(sorted(item['capacity'] for item in report['profiles']),core.CAPS,'one profile per capacity')
        for item in report['profiles']:
            profile_path = (path.parent/item['path']).resolve(strict=True)
            audit.require(profile_path.is_relative_to(base) and profile_path in seen,'manifested raw profile')
            audit.same(core.sha(profile_path),item['sha256'],'profile full byte hash')
            audit.same(profile_path.stat().st_size,item['bytes'],'profile exact byte length')
            audit.same([item['repeats'],item['outside_timing'],item['preparation_in_trace_outside_label']],[1,True,True],'one outside-timing attribution replay')
            row = core.profile_check(core.load(profile_path),item,policy,audit)
            profiles.append(dict(variant=variant,policy=policy,capacity=item['capacity'],**row))
    audit.require(parent_pids.isdisjoint(child_pids),'launcher/native process PID sets disjoint')
    audit.same([sum(row[key] for row in summaries) for key in ('timings','fp32_rows','complete_plan_checks')],
               [40,72,12],'complete raw timing/FP32/plan counts')
    for name in ('device-before.txt','device-cell0.txt','device-cell1.txt','device-after.txt'):
        audit.require('No running processes found in NPU 0' in only(seen,name,audit).read_text(),'NPU0 idle in '+name)
    changes = []
    for cap in core.CAPS:
        old,new = [next(row for row in summary['timing'] if row['capacity']==cap) for summary in summaries]
        pold,pnew = [next(row for row in profiles if row['variant']==variant and row['capacity']==cap) for variant in ('control','resident')]
        changes.append(dict(capacity=cap,control_wall_us=old['wall_us'],resident_wall_us=new['wall_us'],
                            saved_wall_us=old['wall_us']-new['wall_us'],reduction_percent=100*(old['wall_us']-new['wall_us'])/old['wall_us'],
                            control_wall_median_us=old['wall_median_us'],resident_wall_median_us=new['wall_median_us'],
                            median_reduction_percent=100*(old['wall_median_us']-new['wall_median_us'])/old['wall_median_us'],
                            profile_attention_saved_us=pold['attention_sum_us']-pnew['attention_sum_us'],samples_removed=0))
    return dict(qualified=True,assertions=audit.checks,reference_assertions=ref_audit['assertions'],manifest_files=len(seen),
                process_count=2,timings=40,sampled_fp32_rows=72,complete_plan_checks=12,
                same_variant_reference_all_input_output_bytes_and_plans=True,builds=builds,summaries=summaries,
                profile=profiles,consumer_changes=changes,npu0_idle_before_each_and_after=True,
                source=dict(bundle=str(base),reference=str(reference),suite_sha256=core.sha(suite_path),
                            auditor_sha256=core.sha(__file__),pure_helper_auditor_sha256=core.sha(core.__file__)),
                samples_removed=0,stable_performance_gain_established=False,full_trace_measured=False,
                full_model_qualified=False,end_to_end_gain_established=False,
                limitations=['One fresh process per variant in reversed whole-policy order; confounds remain.',
                             'Synthetic two-record sixteen-attention chain, not dynamic/full-model/serving or split-versus-padding evidence.',
                             'Every formal sample and spike retained; four profiles are separate one-replay attribution diagnostics.',
                             'Full output equality applies to the frozen workload; sampled FP32 checks do not prove arbitrary inputs.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('refusing to overwrite audit evidence')
    result = audit_pair(args.bundle,args.reference,core.Audit())
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='ascii')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
