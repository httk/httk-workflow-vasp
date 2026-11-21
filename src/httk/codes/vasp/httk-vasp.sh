#!/usr/bin/env bash

# Native httk VASP Bash API, version 6. Source httk-workflow.sh first.
#
# Every function is one bridge subcommand, and every option of that subcommand is
# available here: the arguments are passed through untouched. Version 2 keeps that
# surface and changes three defaults, in both languages at once:
#
#   * k-point centering starts at Monkhorst-Pack, because the reviewed remedy
#     ladder promotes Gamma as the fix for two k-point failure classes;
#   * httk_vasp_preclean keeps CONTCAR and vasp-run-report.json, which are what a
#     remedy and a restart read; pass --also-remove NAME to delete them anyway;
#   * httk_vasp_rattle_poscar requires --seed or --entropy, because an unseeded
#     retry rattles a structure exactly like the attempt before it.
#
# httk_vasp_remedy_plan and httk_vasp_remedy_apply record the escalation ladder in
# the job state directory by default, so a job with an isolated workdir keeps
# climbing it; httk_vasp_remedy_plan prints the diagnosed problem on stdout and
# takes --directory (the calculation it validates the remedy against) and --policy.
# httk_vasp_remedy_reapply re-applies the INCAR/KPOINTS edits of every remedy in
# that history onto freshly prepared inputs (a new stage), printing them as JSON.
#
# Version 3 adds httk_vasp_remedy_reapply and httk_vasp_report_problems (the
# distinct non-warning diagnostic codes of a run report, one per line).
#
# Version 4 adds httk_vasp_unset_tag TAG [path=INCAR], which removes every
# assignment of TAG so VASP uses its own default; an absent TAG is no error and
# leaves the file untouched. httk_vasp_set_tag TAG null still writes "null".
#
# Version 5 adds httk_vasp_run --soft-stop ionic|electronic, which writes a
# STOPCAR (LSTOP or LABORT) shortly before the attempt's deadline, taken from
# HTTK_WORKFLOW_STOP_WARNING and HTTK_WORKFLOW_DEADLINE. httk_vasp_run exits 0
# (completed), 20 (diagnosed stop), 21 (nonconverged), 22 (process failure),
# 23 (stopped by a STOPCAR, resumable) or 124 (timeout). It also adds
# httk_vasp_soft_stop_mintime [REPORT=vasp-run-report.json], which prints the
# mintime to resume a stopped run with, or exits 1 with no output when there is none.
#
# Version 6 adds httk_vasp_finalize, which finalizes a succeeded job's outputs
# exactly as the Python finalize_job: it cleans OUTCAR, replaces every POTCAR (in
# the workdir and the payload's files/) by POTCAR.summary and POTCAR.sha256, and
# compresses everything in the workdir, configured by the settings vasp.compress,
# vasp.compress_level, vasp.compress_workers, vasp.clean_outcar and
# vasp.summarize_potcar. httk_vasp_clean_outcar [--path OUTCAR] now cleans in
# place, byte-for-byte as httk v1's VASP_CLEAN_OUTCAR; it no longer takes --output.
HTTK_VASP_BASH_API_VERSION=6

_httk_vasp_require_workflow_api() {
    if ! declare -F _httk_workflow_bridge >/dev/null 2>&1; then
        printf 'httk-workflow: source HTTK_WORKFLOW_BASH_API before HTTK_WORKFLOW_VASP_BASH_API\n' >&2
        return 2
    fi
}

httk_vasp_prepare() {
    _httk_vasp_require_workflow_api || return
    _httk_workflow_bridge vasp-prepare "$@"
}

httk_vasp_get_tag() {
    _httk_vasp_require_workflow_api || return
    _httk_workflow_bridge vasp-get-tag "$@"
}

httk_vasp_set_tag() {
    _httk_vasp_require_workflow_api || return
    _httk_workflow_bridge vasp-set-tag "$@"
}

httk_vasp_unset_tag() {
    _httk_vasp_require_workflow_api || return
    _httk_workflow_bridge vasp-unset-tag "$@"
}

httk_vasp_prepare_kpoints() {
    _httk_vasp_require_workflow_api || return
    _httk_workflow_bridge vasp-kpoints "$@"
}

httk_vasp_prepare_potcar() {
    _httk_vasp_require_workflow_api || return
    _httk_workflow_bridge vasp-potcar "$@"
}

httk_vasp_nbands() {
    _httk_vasp_require_workflow_api || return
    _httk_workflow_bridge vasp-nbands "$@"
}

httk_vasp_energy() {
    _httk_vasp_require_workflow_api || return
    _httk_workflow_bridge vasp-energy "$@"
}

httk_vasp_volume() {
    _httk_vasp_require_workflow_api || return
    _httk_workflow_bridge vasp-volume "$@"
}

httk_vasp_potim() {
    _httk_vasp_require_workflow_api || return
    _httk_workflow_bridge vasp-potim "$@"
}

httk_vasp_plane_waves() {
    _httk_vasp_require_workflow_api || return
    _httk_workflow_bridge vasp-plane-waves "$@"
}

httk_vasp_promote_contcar() {
    _httk_vasp_require_workflow_api || return
    _httk_workflow_bridge vasp-promote-contcar "$@"
}

httk_vasp_potcar_summary() {
    _httk_vasp_require_workflow_api || return
    _httk_workflow_bridge vasp-potcar-summary "$@"
}

httk_vasp_clean_outcar() {
    _httk_vasp_require_workflow_api || return
    _httk_workflow_bridge vasp-clean-outcar "$@"
}

httk_vasp_finalize() {
    _httk_vasp_require_workflow_api || return
    local name value status arguments=()
    value=$(httk_workflow_setting vasp.compress zst) || return
    # A JSON null (or the string "null") means unset, as in finalize_job: the default applies.
    arguments+=(--compress "$([ "$value" = null ] && printf zst || printf '%s' "$value")")
    for name in level workers; do
        value=$(httk_workflow_setting "vasp.compress_$name") || {
            status=$?
            [ "$status" -eq 1 ] && continue
            return "$status"
        }
        [ "$value" = null ] || arguments+=("--$name" "$value")
    done
    for name in clean_outcar summarize_potcar; do
        value=$(httk_workflow_setting "vasp.$name" true) || return
        [ "$value" = null ] && value=true
        case "${value,,}" in
            true | yes | 1) ;;
            false | no | 0) arguments+=("$([ "$name" = clean_outcar ] && printf -- --no-clean-outcar || printf -- --keep-potcar)") ;;
            *)
                printf 'httk-workflow: setting vasp.%s must be true or false (or yes/no/1/0), got %s\n' "$name" "$value" >&2
                return 2
                ;;
        esac
    done
    _httk_workflow_bridge vasp-finalize "${arguments[@]}" --compress-root "${HTTK_WORKFLOW_WORKDIR:?}" \
        "$HTTK_WORKFLOW_WORKDIR" "${HTTK_WORKFLOW_JOB_DIR:?}/files"
}

httk_vasp_preclean() {
    _httk_vasp_require_workflow_api || return
    _httk_workflow_bridge vasp-preclean "$@"
}

httk_vasp_normalize_poscar() {
    _httk_vasp_require_workflow_api || return
    _httk_workflow_bridge vasp-normalize-poscar "$@"
}

httk_vasp_scale_poscar() {
    _httk_vasp_require_workflow_api || return
    _httk_workflow_bridge vasp-scale-poscar "$@"
}

httk_vasp_rattle_poscar() {
    _httk_vasp_require_workflow_api || return
    _httk_workflow_bridge vasp-rattle-poscar "$@"
}

httk_vasp_run() {
    _httk_vasp_require_workflow_api || return
    _httk_workflow_bridge vasp-run "$@"
}

httk_vasp_soft_stop_mintime() {
    _httk_vasp_require_workflow_api || return
    _httk_workflow_bridge vasp-soft-stop-mintime "$@"
}

httk_vasp_diagnose() {
    _httk_vasp_require_workflow_api || return
    _httk_workflow_bridge vasp-diagnose "$@"
}

httk_vasp_report_problems() {
    _httk_vasp_require_workflow_api || return
    _httk_workflow_bridge vasp-report-problems "$@"
}

httk_vasp_remedy_plan() {
    _httk_vasp_require_workflow_api || return
    _httk_workflow_bridge vasp-remedy-plan "$@"
}

httk_vasp_remedy_apply() {
    _httk_vasp_require_workflow_api || return
    _httk_workflow_bridge vasp-remedy-apply "$@"
}

httk_vasp_remedy_reapply() {
    _httk_vasp_require_workflow_api || return
    _httk_workflow_bridge vasp-remedy-reapply "$@"
}
