import re
import duckdb
import logging
import datetime
import pandas as pd
import threading
from typing import Union, Dict, List

from ...utils.log import L
from ..base import DatabaseHandler
from .. import build_config, DatabaseTypes

logger = L.get_logger("frozen")

class DuckDBHandler(DatabaseHandler):
    """DuckDB database handler"""
    
    _instance = None
    _lock = threading.RLock()
    _initialized = False
    
    def __new__(cls, db_path: str = None):
        # singleton pattern for thread safety
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(DuckDBHandler, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, db_path: str = None):
        # ensure initialization only once
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self.config = build_config(DatabaseTypes.DUCKDB.value)
                    self.db = self.config.data_path if db_path is None else db_path
                    self.__class__._initialized = True
    
    def init_db(self):
        self.config.connect()
        self._query("SELECT 1")
    
    def _check_table_exists(self, table_name=""):
        query_str = f"""
                    SELECT * 
                    FROM information_schema.tables 
                    WHERE table_name='{table_name}'
                    """
        res = self._query(query_str, fmt="list")
        return res
    
    def _check_table_empty(self, table_name):
        query_str = f"SELECT COUNT(*) FROM {table_name}"
        result = self._query(query_str, fmt="tuple")
        res = result[0] == 0
        return res
    
    def _check_data_exists(self, table_name, data_str, source_str=None):
        if source_str:
            query_str = f"SELECT * FROM {table_name} WHERE ticker='{data_str}' and source='{source_str}'"
        else:
            if "." in data_str:  # ticker
                query_str = f"SELECT * FROM {table_name} WHERE ticker='{data_str}'"
            else:  # trade_date
                query_str = f"SELECT * FROM {table_name} WHERE trade_date='{data_str}'"
        result = self._query(query_str, fmt="list")
        return result

    def _get_table_ticker(self, table_name):
        """Get all unique tickers from a table"""
        query_str = f"SELECT DISTINCT ticker FROM {table_name}"
        result = self._query(query_str, fmt="dataframe")
        res = result["ticker"].tolist()
        return res
    
    def _get_table_date(self, table_name, latest=False) -> Union[pd.DataFrame, pd.Timestamp]:
        """Get the latest date for each ticker in a table"""
        date_col = "ann_date" if table_name == "stock_dividend" else "trade_date"
        query_str = f"SELECT ticker, MAX({date_col}) AS max_date FROM {table_name} GROUP BY ticker"
        table_date = self._query(query_str, fmt="dataframe")
        if latest:
            table_date = table_date["max_date"].max()
        return table_date
    
    def _get_ticker_date(self, table_date, ticker, shift=0) -> str:
        """Get the latest date for a ticker with optional date shift"""
        ticker_date = table_date[table_date["ticker"]==ticker]["max_date"].iloc[0]
        if isinstance(ticker_date, pd.Timestamp):
            ticker_date = (ticker_date + datetime.timedelta(shift)).strftime("%Y%m%d")
        elif isinstance(ticker_date, str):
            ticker_date = (datetime.datetime.strptime(ticker_date, "%Y%m%d") + datetime.timedelta(shift)).strftime("%Y%m%d")
        else:
            raise ValueError(f"Invalid ticker date type: {type(ticker_date)}")
        return ticker_date
    
    def _get_latest_ticker_date(self, table_name, ticker):
        """Get the next day after the latest date for a ticker"""
        table_date = self._get_table_date(table_name, latest=False)
        start_date = self._get_ticker_date(table_date, ticker, shift=1)
        return start_date

    def _get_table_earliest_date(self, table_name) -> pd.DataFrame:
        """Get the earliest date for each ticker in the table"""
        date_col = "ann_date" if table_name == "stock_dividend" else "trade_date"
        query_str = f"SELECT ticker, MIN({date_col}) AS min_date FROM {table_name} GROUP BY ticker"
        table_earliest_date = self._query(query_str, fmt="dataframe")
        return table_earliest_date

    def _get_earliest_ticker_date(self, table_name, ticker):
        """Get the earliest date for a specific ticker in the table"""
        table_earliest_date = self._get_table_earliest_date(table_name)
        if ticker not in table_earliest_date["ticker"].values:
            return None
        earliest_date = table_earliest_date[table_earliest_date["ticker"]==ticker]["min_date"].iloc[0]
        if isinstance(earliest_date, pd.Timestamp):
            earliest_date = earliest_date.strftime("%Y%m%d")
        return earliest_date

    def _insert_df_to_table(self, df, table_name):
        with self._lock:
            with duckdb.connect(self.db) as conn:
                try:
                    df.to_sql(table_name, conn, chunksize=1000000, if_exists="append", index=False)
                except Exception as e:
                    # Handle transaction rollback error - common with pandas to_sql and DuckDB
                    if "cannot rollback - no transaction is active" in str(e):
                        # Try alternative insertion method
                        try:
                            conn.register('temp_df', df)
                            conn.execute(f"INSERT INTO {table_name} SELECT * FROM temp_df")
                            conn.unregister('temp_df')
                        except Exception as e2:
                            logger.error(f"Failed to insert data to table {table_name} using alternative method: {e2}")
                            raise e2
                    else:
                        logger.error(f"Failed to insert data to table {table_name}: {e}")
                        raise e

    def _delete_table(self, table_name):
        try:
            self._query(f"DROP TABLE IF EXISTS {table_name}")
            logger.critical(f"Table {table_name} has been dropped.")
        except Exception as e:
            logger.error(f"Failed to drop table {table_name}: {e}")
    
    def _clear_table(self, table_name):
        """Clear all data in the table but keep the table structure."""
        try:
            self._query(f"DELETE FROM {table_name}")
            logger.info(f"Table {table_name} has been cleared.")
        except Exception as e:
            logger.error(f"Failed to clear table {table_name}: {e}")
    
    def _copy_table(self, target_path, source_table, target_table=None):
        """Copy table to another specified database."""
        if target_table is None:
            target_table = source_table
        
        query_str = f"""
            ATTACH '{target_path}' AS target_db;
            CREATE TABLE target_db.{target_table} AS SELECT * FROM {source_table};
            DETACH target_db;
            """
        try:
            self._query(query_str)
            logger.info(f"Table {source_table} has been copied to target database.")
        except Exception as e:
            logger.error(f"Failed to copy table {source_table}: {e}")
    
    def _query(self, query_str: Union[str, Dict], params: tuple = None, fmt: str = None, read_only: bool = False):
        """
        Execute query and return results in specified format with parameter support.
        
        Args:
            query_str: SQL query string or dictionary
            params: Query parameters as a tuple (default: None)
            fmt: Return format ("dataframe", "list", or "tuple")
            read_only: Read-only mode flag
            
        Returns:
            Query results in specified format or None for non-SELECT queries
        """

        if fmt is None:
            fmt = "dataframe"
        if fmt not in ["dataframe", "list", "tuple"]:
            raise ValueError(f"duckdb only supports 'dataframe', 'list' or 'tuple' format, got '{fmt}'")
        
        # Ensure params is a tuple if provided
        if params is not None and not isinstance(params, tuple):
            params = tuple(params)
        
        # use class-level lock to ensure thread safety (all instances share the same lock)
        with self._lock:
            with duckdb.connect(self.db, read_only=read_only) as conn:
                cursor = conn.cursor()
                
                # Execute with parameters if provided
                if params:
                    cursor.execute(query_str, params)
                else:
                    cursor.execute(query_str)
                    
                # use regex to identify SELECT queries (including WITH CTEs)
                if not re.match(r'^\s*(SELECT|WITH)', query_str, re.IGNORECASE):
                    return None
                try:
                    if fmt == "dataframe":
                        return cursor.fetch_df()
                    elif fmt == "list":
                        return cursor.fetchall()
                    elif fmt == "tuple":
                        return cursor.fetchone()
                    else:
                        return None  # Should never reach here due to validation above
                except Exception as e:
                    print(f"Error fetching data: {e}")
                    return None

    def create_volume_price_table(self, table_name):
        self._query(f"""
                    CREATE TABLE IF NOT EXISTS {table_name}
                    (
                        ticker VARCHAR,
                        trade_date TIMESTAMP,
                        open DOUBLE,
                        high DOUBLE,
                        low DOUBLE,
                        close DOUBLE,
                        pre_close DOUBLE,
                        change DOUBLE,
                        pct_chg DOUBLE,
                        volume DOUBLE,
                        amount DOUBLE,
                        PRIMARY KEY (ticker, trade_date)
                    )
                    """)
    
    def create_stock_limit_table(self, table_name):
        self._query(f"""
                    CREATE TABLE IF NOT EXISTS {table_name}
                    (
                        trade_date TIMESTAMP,
                        ticker VARCHAR,
                        up_limit DOUBLE,
                        down_limit DOUBLE,
                        PRIMARY KEY (ticker, trade_date)
                    )
                    """)
    
    def create_stock_fundamental_table(self, table_name):
        self._query(f"""
                    CREATE TABLE IF NOT EXISTS {table_name}
                    (
                        ticker VARCHAR,
                        trade_date TIMESTAMP,
                        turnover_rate DOUBLE,
                        turnover_rate_f DOUBLE,
                        volume_ratio DOUBLE,
                        pe DOUBLE,
                        pe_ttm DOUBLE,
                        pb DOUBLE,
                        ps DOUBLE,
                        ps_ttm DOUBLE,
                        dv_ratio DOUBLE,
                        dv_ttm DOUBLE,
                        total_share DOUBLE,
                        float_share DOUBLE,
                        free_share DOUBLE,
                        total_mv DOUBLE,
                        circ_mv DOUBLE,
                        PRIMARY KEY (ticker, trade_date)
                    )
                    """)
    
    def create_stock_dividend_table(self, table_name):
        self._query(f"""
                    CREATE TABLE IF NOT EXISTS {table_name}
                    (
                        ticker VARCHAR,
                        end_date TIMESTAMP,
                        ann_date TIMESTAMP,
                        div_proc VARCHAR,
                        stk_div DOUBLE,
                        stk_bo_rate DOUBLE,
                        stk_co_rate DOUBLE,
                        cash_div DOUBLE,
                        cash_div_tax DOUBLE,
                        record_date TIMESTAMP,
                        ex_date TIMESTAMP,
                        pay_date TIMESTAMP,
                        div_listdate TIMESTAMP,
                        imp_ann_date TIMESTAMP,
                        base_date TIMESTAMP,
                        base_share DOUBLE,
                        PRIMARY KEY (ticker, end_date, ann_date, div_proc)
                    )
                    """)
    
    def create_stock_suspend_table(self, table_name):
        self._query(f"""
                    CREATE TABLE IF NOT EXISTS {table_name}
                    (
                        ticker VARCHAR,
                        trade_date TIMESTAMP,
                        suspend_timing VARCHAR,
                        suspend_type VARCHAR,
                    )
                    """)
    
    def create_stock_basic_table(self, table_name):
        self._query(f"""
                    CREATE TABLE IF NOT EXISTS {table_name}
                    (
                        ticker VARCHAR PRIMARY KEY,
                        symbol VARCHAR,
                        name VARCHAR,
                        area VARCHAR,
                        industry VARCHAR,
                        fullname VARCHAR,
                        enname VARCHAR,
                        cnspell VARCHAR,
                        market VARCHAR,
                        exchange VARCHAR,
                        curr_type VARCHAR,
                        list_status VARCHAR,
                        list_date TIMESTAMP,
                        delist_date TIMESTAMP,
                        is_hs VARCHAR,
                        act_name VARCHAR,
                        act_ent_type VARCHAR
                    )
                    """)

    def create_stock_income_table(self, table_name):
        self._query(f"""
                    CREATE TABLE IF NOT EXISTS {table_name}
                    (
                        ticker VARCHAR,
                        ann_date TIMESTAMP,
                        f_ann_date TIMESTAMP,
                        end_date TIMESTAMP,
                        report_type VARCHAR,
                        comp_type VARCHAR,
                        end_type VARCHAR,
                        basic_eps DOUBLE,
                        diluted_eps DOUBLE,
                        total_revenue DOUBLE,
                        revenue DOUBLE,
                        int_income DOUBLE,
                        prem_earned DOUBLE,
                        comm_income DOUBLE,
                        n_commis_income DOUBLE,
                        n_oth_income DOUBLE,
                        n_oth_b_income DOUBLE,
                        prem_income DOUBLE,
                        out_prem DOUBLE,
                        une_prem_reser DOUBLE,
                        reins_income DOUBLE,
                        n_sec_tb_income DOUBLE,
                        n_sec_uw_income DOUBLE,
                        n_asset_mg_income DOUBLE,
                        oth_b_income DOUBLE,
                        fv_value_chg_gain DOUBLE,
                        invest_income DOUBLE,
                        ass_invest_income DOUBLE,
                        forex_gain DOUBLE,
                        total_cogs DOUBLE,
                        oper_cost DOUBLE,
                        int_exp DOUBLE,
                        comm_exp DOUBLE,
                        biz_tax_surchg DOUBLE,
                        sell_exp DOUBLE,
                        admin_exp DOUBLE,
                        fin_exp DOUBLE,
                        assets_impair_loss DOUBLE,
                        prem_refund DOUBLE,
                        compens_payout DOUBLE,
                        reser_insur_liab DOUBLE,
                        div_payt DOUBLE,
                        reins_exp DOUBLE,
                        oper_exp DOUBLE,
                        compens_payout_refu DOUBLE,
                        insur_reser_refu DOUBLE,
                        reins_cost_refund DOUBLE,
                        other_bus_cost DOUBLE,
                        operate_profit DOUBLE,
                        non_oper_income DOUBLE,
                        non_oper_exp DOUBLE,
                        nca_disploss DOUBLE,
                        total_profit DOUBLE,
                        income_tax DOUBLE,
                        n_income DOUBLE,
                        n_income_attr_p DOUBLE,
                        minority_gain DOUBLE,
                        oth_compr_income DOUBLE,
                        t_compr_income DOUBLE,
                        compr_inc_attr_p DOUBLE,
                        compr_inc_attr_m_s DOUBLE,
                        ebit DOUBLE,
                        ebitda DOUBLE,
                        insurance_exp DOUBLE,
                        undist_profit DOUBLE,
                        distable_profit DOUBLE,
                        rd_exp DOUBLE,
                        fin_exp_int_exp DOUBLE,
                        fin_exp_int_inc DOUBLE,
                        transfer_surplus_rese DOUBLE,
                        transfer_housing_imprest DOUBLE,
                        transfer_oth DOUBLE,
                        adj_lossgain DOUBLE,
                        withdra_legal_surplus DOUBLE,
                        withdra_legal_pubfund DOUBLE,
                        withdra_biz_devfund DOUBLE,
                        withdra_rese_fund DOUBLE,
                        withdra_oth_ersu DOUBLE,
                        workers_welfare DOUBLE,
                        distr_profit_shrhder DOUBLE,
                        prfshare_payable_dvd DOUBLE,
                        comshare_payable_dvd DOUBLE,
                        capit_comstock_div DOUBLE,
                        net_after_nr_lp_correct DOUBLE,
                        credit_impa_loss DOUBLE,
                        net_expo_hedging_benefits DOUBLE,
                        oth_impair_loss_assets DOUBLE,
                        total_opcost DOUBLE,
                        amodcost_fin_assets DOUBLE,
                        oth_income DOUBLE,
                        asset_disp_income DOUBLE,
                        continued_net_profit DOUBLE,
                        end_net_profit DOUBLE,
                        update_flag VARCHAR,
                        PRIMARY KEY (ticker, f_ann_date, end_date, report_type)
                    )
                    """)
    
    def create_stock_balancesheet_table(self, table_name):
        self._query(f"""
                    CREATE TABLE IF NOT EXISTS {table_name}
                    (
                        ticker VARCHAR,
                        ann_date TIMESTAMP,
                        f_ann_date TIMESTAMP,
                        end_date TIMESTAMP,
                        report_type VARCHAR,
                        comp_type VARCHAR,
                        end_type VARCHAR,
                        total_share DOUBLE,
                        cap_rese DOUBLE,
                        undistr_porfit DOUBLE,
                        surplus_rese DOUBLE,
                        special_rese DOUBLE,
                        money_cap DOUBLE,
                        trad_asset DOUBLE,
                        notes_receiv DOUBLE,
                        accounts_receiv DOUBLE,
                        oth_receiv DOUBLE,
                        prepayment DOUBLE,
                        div_receiv DOUBLE,
                        int_receiv DOUBLE,
                        inventories DOUBLE,
                        amor_exp DOUBLE,
                        nca_within_1y DOUBLE,
                        sett_rsrv DOUBLE,
                        loanto_oth_bank_fi DOUBLE,
                        premium_receiv DOUBLE,
                        reinsur_receiv DOUBLE,
                        reinsur_res_receiv DOUBLE,
                        pur_resale_fa DOUBLE,
                        oth_cur_assets DOUBLE,
                        total_cur_assets DOUBLE,
                        fa_avail_for_sale DOUBLE,
                        htm_invest DOUBLE,
                        lt_eqt_invest DOUBLE,
                        invest_real_estate DOUBLE,
                        time_deposits DOUBLE,
                        oth_assets DOUBLE,
                        lt_rec DOUBLE,
                        fix_assets DOUBLE,
                        cip DOUBLE,
                        const_materials DOUBLE,
                        fixed_assets_disp DOUBLE,
                        produc_bio_assets DOUBLE,
                        oil_and_gas_assets DOUBLE,
                        intan_assets DOUBLE,
                        r_and_d DOUBLE,
                        goodwill DOUBLE,
                        lt_amor_exp DOUBLE,
                        defer_tax_assets DOUBLE,
                        decr_in_disbur DOUBLE,
                        oth_nca DOUBLE,
                        total_nca DOUBLE,
                        cash_reser_cb DOUBLE,
                        depos_in_oth_bfi DOUBLE,
                        prec_metals DOUBLE,
                        deriv_assets DOUBLE,
                        rr_reins_une_prem DOUBLE,
                        rr_reins_outstd_cla DOUBLE,
                        rr_reins_lins_liab DOUBLE,
                        rr_reins_lthins_liab DOUBLE,
                        refund_depos DOUBLE,
                        ph_pledge_loans DOUBLE,
                        refund_cap_depos DOUBLE,
                        indep_acct_assets DOUBLE,
                        client_depos DOUBLE,
                        client_prov DOUBLE,
                        transac_seat_fee DOUBLE,
                        invest_as_receiv DOUBLE,
                        total_assets DOUBLE,
                        lt_borr DOUBLE,
                        st_borr DOUBLE,
                        cb_borr DOUBLE,
                        depos_ib_deposits DOUBLE,
                        loan_oth_bank DOUBLE,
                        trading_fl DOUBLE,
                        notes_payable DOUBLE,
                        acct_payable DOUBLE,
                        adv_receipts DOUBLE,
                        sold_for_repur_fa DOUBLE,
                        comm_payable DOUBLE,
                        payroll_payable DOUBLE,
                        taxes_payable DOUBLE,
                        int_payable DOUBLE,
                        div_payable DOUBLE,
                        oth_payable DOUBLE,
                        acc_exp DOUBLE,
                        deferred_inc DOUBLE,
                        st_bonds_payable DOUBLE,
                        payable_to_reinsurer DOUBLE,
                        rsrv_insur_cont DOUBLE,
                        acting_trading_sec DOUBLE,
                        acting_uw_sec DOUBLE,
                        non_cur_liab_due_1y DOUBLE,
                        oth_cur_liab DOUBLE,
                        total_cur_liab DOUBLE,
                        bond_payable DOUBLE,
                        lt_payable DOUBLE,
                        specific_payables DOUBLE,
                        estimated_liab DOUBLE,
                        defer_tax_liab DOUBLE,
                        defer_inc_non_cur_liab DOUBLE,
                        oth_ncl DOUBLE,
                        total_ncl DOUBLE,
                        depos_oth_bfi DOUBLE,
                        deriv_liab DOUBLE,
                        depos DOUBLE,
                        agency_bus_liab DOUBLE,
                        oth_liab DOUBLE,
                        prem_receiv_adva DOUBLE,
                        depos_received DOUBLE,
                        ph_invest DOUBLE,
                        reser_une_prem DOUBLE,
                        reser_outstd_claims DOUBLE,
                        reser_lins_liab DOUBLE,
                        reser_lthins_liab DOUBLE,
                        indept_acc_liab DOUBLE,
                        pledge_borr DOUBLE,
                        indem_payable DOUBLE,
                        policy_div_payable DOUBLE,
                        total_liab DOUBLE,
                        treasury_share DOUBLE,
                        ordin_risk_reser DOUBLE,
                        forex_differ DOUBLE,
                        invest_loss_unconf DOUBLE,
                        minority_int DOUBLE,
                        total_hldr_eqy_exc_min_int DOUBLE,
                        total_hldr_eqy_inc_min_int DOUBLE,
                        total_liab_hldr_eqy DOUBLE,
                        lt_payroll_payable DOUBLE,
                        oth_comp_income DOUBLE,
                        oth_eqt_tools DOUBLE,
                        oth_eqt_tools_p_shr DOUBLE,
                        lending_funds DOUBLE,
                        acc_receivable DOUBLE,
                        st_fin_payable DOUBLE,
                        payables DOUBLE,
                        hfs_assets DOUBLE,
                        hfs_sales DOUBLE,
                        cost_fin_assets DOUBLE,
                        fair_value_fin_assets DOUBLE,
                        cip_total DOUBLE,
                        oth_pay_total DOUBLE,
                        long_pay_total DOUBLE,
                        debt_invest DOUBLE,
                        oth_debt_invest DOUBLE,
                        oth_eq_invest DOUBLE,
                        oth_illiq_fin_assets DOUBLE,
                        oth_eq_ppbond DOUBLE,
                        receiv_financing DOUBLE,
                        use_right_assets DOUBLE,
                        lease_liab DOUBLE,
                        contract_assets DOUBLE,
                        contract_liab DOUBLE,
                        accounts_receiv_bill DOUBLE,
                        accounts_pay DOUBLE,
                        oth_rcv_total DOUBLE,
                        fix_assets_total DOUBLE,
                        update_flag VARCHAR,
                        PRIMARY KEY (ticker, f_ann_date, end_date, report_type)
                    )
                    """)
    
    def create_stock_cashflow_table(self, table_name):
        self._query(f"""
                    CREATE TABLE IF NOT EXISTS {table_name}
                    (
                        ticker VARCHAR,
                        ann_date TIMESTAMP,
                        f_ann_date TIMESTAMP,
                        end_date TIMESTAMP,
                        comp_type VARCHAR,
                        report_type VARCHAR,
                        end_type VARCHAR,
                        net_profit DOUBLE,
                        finan_exp DOUBLE,
                        c_fr_sale_sg DOUBLE,
                        recp_tax_rends DOUBLE,
                        n_depos_incr_fi DOUBLE,
                        n_incr_loans_cb DOUBLE,
                        n_inc_borr_oth_fi DOUBLE,
                        prem_fr_orig_contr DOUBLE,
                        n_incr_insured_dep DOUBLE,
                        n_reinsur_prem DOUBLE,
                        n_incr_disp_tfa DOUBLE,
                        ifc_cash_incr DOUBLE,
                        n_incr_disp_faas DOUBLE,
                        n_incr_loans_oth_bank DOUBLE,
                        n_cap_incr_repur DOUBLE,
                        c_fr_oth_operate_a DOUBLE,
                        c_inf_fr_operate_a DOUBLE,
                        c_paid_goods_s DOUBLE,
                        c_paid_to_for_empl DOUBLE,
                        c_paid_for_taxes DOUBLE,
                        n_incr_clt_loan_adv DOUBLE,
                        n_incr_dep_cbob DOUBLE,
                        c_pay_claims_orig_inco DOUBLE,
                        pay_handling_chrg DOUBLE,
                        pay_comm_insur_plcy DOUBLE,
                        oth_cash_pay_oper_act DOUBLE,
                        st_cash_out_act DOUBLE,
                        n_cashflow_act DOUBLE,
                        oth_recp_ral_inv_act DOUBLE,
                        c_disp_withdrwl_invest DOUBLE,
                        c_recp_return_invest DOUBLE,
                        n_recp_disp_fiolta DOUBLE,
                        n_recp_disp_sobu DOUBLE,
                        stot_inflows_inv_act DOUBLE,
                        c_pay_acq_const_fiolta DOUBLE,
                        c_paid_invest DOUBLE,
                        n_disp_subs_oth_biz DOUBLE,
                        oth_pay_ral_inv_act DOUBLE,
                        n_incr_pledge_loan DOUBLE,
                        stot_out_inv_act DOUBLE,
                        n_cashflow_inv_act DOUBLE,
                        c_recp_borrow DOUBLE,
                        proc_issue_bonds DOUBLE,
                        oth_cash_recp_ral_fnc_act DOUBLE,
                        stot_cash_in_fnc_act DOUBLE,
                        free_cashflow DOUBLE,
                        c_prepay_amt_borr DOUBLE,
                        c_pay_dist_dpcp_int_exp DOUBLE,
                        incl_dvd_profit_paid_sc_ms DOUBLE,
                        oth_cashpay_ral_fnc_act DOUBLE,
                        stot_cashout_fnc_act DOUBLE,
                        n_cash_flows_fnc_act DOUBLE,
                        eff_fx_flu_cash DOUBLE,
                        n_incr_cash_cash_equ DOUBLE,
                        c_cash_equ_beg_period DOUBLE,
                        c_cash_equ_end_period DOUBLE,
                        c_recp_cap_contrib DOUBLE,
                        incl_cash_rec_saims DOUBLE,
                        uncon_invest_loss DOUBLE,
                        prov_depr_assets DOUBLE,
                        depr_fa_coga_dpba DOUBLE,
                        amort_intang_assets DOUBLE,
                        lt_amort_deferred_exp DOUBLE,
                        decr_deferred_exp DOUBLE,
                        incr_acc_exp DOUBLE,
                        loss_disp_fiolta DOUBLE,
                        loss_scr_fa DOUBLE,
                        loss_fv_chg DOUBLE,
                        invest_loss DOUBLE,
                        decr_def_inc_tax_assets DOUBLE,
                        incr_def_inc_tax_liab DOUBLE,
                        decr_inventories DOUBLE,
                        decr_oper_payable DOUBLE,
                        incr_oper_payable DOUBLE,
                        others DOUBLE,
                        im_net_cashflow_oper_act DOUBLE,
                        conv_debt_into_cap DOUBLE,
                        conv_copbonds_due_within_1y DOUBLE,
                        fa_fnc_leases DOUBLE,
                        im_n_incr_cash_equ DOUBLE,
                        net_dism_capital_add DOUBLE,
                        net_cash_rece_sec DOUBLE,
                        credit_impa_loss DOUBLE,
                        use_right_asset_dep DOUBLE,
                        oth_loss_asset DOUBLE,
                        end_bal_cash DOUBLE,
                        beg_bal_cash DOUBLE,
                        end_bal_cash_equ DOUBLE,
                        beg_bal_cash_equ DOUBLE,
                        update_flag VARCHAR,
                        PRIMARY KEY (ticker, f_ann_date, end_date, report_type)
                    )
                    """)

    def create_trade_calendar_table(self, table_name):
        self._query(f"""
                    CREATE TABLE IF NOT EXISTS {table_name}
                    (
                        exchange VARCHAR,
                        cal_date TIMESTAMP,
                        is_open UTINYINT,
                        pretrade_date TIMESTAMP,
                    )
                    """)
    
    def create_industry_mapping_table(self, table_name):
                self._query(f"""
                    CREATE TABLE IF NOT EXISTS {table_name}
                    (
                        l1_code VARCHAR,
                        l1_name VARCHAR,
                        l2_code VARCHAR,
                        l2_name VARCHAR,
                        l3_code VARCHAR,
                        l3_name VARCHAR,
                        ticker VARCHAR,
                        name VARCHAR,
                        in_date TIMESTAMP,
                        out_date TIMESTAMP,
                        is_new VARCHAR,
                        source VARCHAR,
                        PRIMARY KEY (l1_code, l2_code, l3_code, ticker)
                    )
                    """)
    
    def create_index_weight_table(self, table_name):
        self._query(f"""
                    CREATE TABLE IF NOT EXISTS {table_name}
                    (
                        ticker VARCHAR,
                        con_code VARCHAR,
                        trade_date TIMESTAMP,
                        weight DOUBLE,
                        PRIMARY KEY (ticker, con_code, trade_date)
                    )
                    """)
    
    def create_cb_basic_table(self, table_name):
        self._query(f"""
                    CREATE TABLE IF NOT EXISTS {table_name}
                    (
                        ticker VARCHAR,
                        bond_full_name VARCHAR,
                        bond_short_name VARCHAR,
                        cb_code VARCHAR,
                        stk_code VARCHAR,
                        stk_short_name VARCHAR,
                        maturity DOUBLE,
                        par DOUBLE,
                        issue_price DOUBLE,
                        issue_size DOUBLE,
                        remain_size DOUBLE,
                        value_date TIMESTAMP,
                        maturity_date TIMESTAMP,
                        rate_type VARCHAR,
                        coupon_rate DOUBLE,
                        add_rate DOUBLE,
                        pay_per_year INT,
                        list_date TIMESTAMP,
                        delist_date TIMESTAMP,
                        exchange VARCHAR,
                        conv_start_date TIMESTAMP,
                        conv_end_date TIMESTAMP,
                        conv_stop_date TIMESTAMP,
                        first_conv_price DOUBLE,
                        conv_price DOUBLE,
                        rate_clause VARCHAR,
                        put_clause VARCHAR,
                        maturity_put_price DOUBLE,
                        call_clause VARCHAR,
                        reset_clause VARCHAR,
                        conv_clause VARCHAR,
                        guarantor VARCHAR,
                        guarantee_type VARCHAR,
                        issue_rating VARCHAR,
                        newest_rating VARCHAR,
                        rating_comp VARCHAR
                    )
                    """)
        
    def create_cb_daily_table(self, table_name):
        self._query(f"""
                    CREATE TABLE IF NOT EXISTS {table_name}
                    (
                        ticker VARCHAR,
                        trade_date TIMESTAMP,
                        pre_close DOUBLE,
                        open DOUBLE,
                        high DOUBLE,
                        low DOUBLE,
                        close DOUBLE,
                        change DOUBLE,
                        pct_chg	DOUBLE,
                        volume DOUBLE,
                        amount DOUBLE,
                        bond_value DOUBLE,
                        bond_over_rate DOUBLE,
                        cb_value DOUBLE,
                        cb_over_rate DOUBLE
                    )
                    """)
    
    def add_comments(self, table_name):
        mapping = financial_report_mapping.get(table_name, {})
        for key, value in mapping.items():
            self._query(f"COMMENT ON COLUMN {table_name}.{key} IS '{value}'")
    
    def create_table_from_dataframe(self, df: pd.DataFrame, table_name: str, primary_keys: List[str] = None):
        """Create table based on DataFrame column structure"""
        
        def pandas_to_duckdb_type(dtype):
            """Convert pandas dtype to DuckDB type"""
            if pd.api.types.is_integer_dtype(dtype):
                return "BIGINT"
            elif pd.api.types.is_float_dtype(dtype):
                return "DOUBLE"
            elif pd.api.types.is_bool_dtype(dtype):
                return "BOOLEAN"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                return "TIMESTAMP"
            else:
                return "VARCHAR"
        
        # Generate column definitions
        columns = []
        for col in df.columns:
            col_type = pandas_to_duckdb_type(df[col].dtype)
            columns.append(f"{col} {col_type}")
        
        # Add primary key constraint if specified
        primary_key_clause = ""
        if primary_keys:
            primary_key_clause = f", PRIMARY KEY ({', '.join(primary_keys)})"
        
        # Create table SQL
        create_sql = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                {', '.join(columns)}{primary_key_clause}
            )
        """
        
        self._query(create_sql)
        logger.info(f"Created table {table_name} with {len(columns)} columns")

    # CSV Loader specific methods
    def batch_insert_dataframe(self, df: pd.DataFrame, table_name: str, batch_size: int = 10000):
        """Insert DataFrame data in batches"""
        total_rows = len(df)
        
        for i in range(0, total_rows, batch_size):
            batch_df = df.iloc[i:i + batch_size]
            
            with self._lock:
                with duckdb.connect(self.db) as conn:
                    # Use register method for batch insertion
                    conn.register("temp_batch", batch_df)
                    conn.execute(f"INSERT INTO {table_name} SELECT * FROM temp_batch")
                    conn.unregister("temp_batch")
        
        logger.info(f"Inserted {total_rows} rows into {table_name} in batches of {batch_size}")
    
    def init_csv_metadata_tables(self):
        """Initialize metadata tables for CSV loader"""
        # File processing metadata table
        self._query("""
            CREATE TABLE IF NOT EXISTS csv_file_metadata (
                id INTEGER,
                file_path VARCHAR NOT NULL UNIQUE,
                table_name VARCHAR NOT NULL,
                year VARCHAR,
                month VARCHAR,
                exchange VARCHAR,
                symbol VARCHAR,
                file_size BIGINT,
                row_count BIGINT,
                file_hash VARCHAR,
                processed_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Table schema registry
        self._query("""
            CREATE TABLE IF NOT EXISTS csv_table_schemas (
                table_name VARCHAR PRIMARY KEY,
                column_info JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        logger.info("CSV metadata tables initialized")
    
    def store_file_metadata(self, metadata: Dict):
        """Store file processing metadata"""
        self._query("""
            INSERT INTO csv_file_metadata 
            (file_path, table_name, year, month, exchange, symbol, file_size, row_count, file_hash, processed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (file_path) DO NOTHING
        """, (
            metadata["file_path"], metadata["table_name"], metadata.get("year"),
            metadata.get("month"), metadata.get("exchange"), metadata.get("symbol"),
            metadata.get("file_size", 0), metadata.get("row_count", 0),
            metadata.get("file_hash", ""), metadata.get("processed_at")
        ))
    
    def check_file_processed(self, file_path: str, file_hash: str) -> bool:
        """Check if file has already been processed"""
        result = self._query("""
            SELECT file_hash FROM csv_file_metadata 
            WHERE file_path = ? AND file_hash = ?
        """, (file_path, file_hash), fmt="tuple")
        
        return result is not None
    
    def get_csv_loading_summary(self) -> pd.DataFrame:
        """Get summary of loaded CSV files and tables"""
        return self._query("""
            SELECT 
                table_name,
                COUNT(*) as file_count,
                SUM(row_count) as total_rows,
                SUM(file_size) as total_size_bytes,
                MIN(processed_at) as first_loaded,
                MAX(processed_at) as last_loaded
            FROM csv_file_metadata 
            GROUP BY table_name
            ORDER BY total_rows DESC
        """, fmt="dataframe")
    
    def list_csv_tables(self) -> List[str]:
        """List all CSV data tables (excluding metadata tables)"""
        result = self._query("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_name NOT IN ('csv_file_metadata', 'csv_table_schemas')
            AND table_name NOT LIKE 'init%'
            ORDER BY table_name
        """, fmt="list")
        return [row[0] for row in result] if result else []


stock_income_mapping = {
    "ticker": "股票代码",
    "ann_date": "公告日期",
    "f_ann_date": "实际公告日期",
    "end_date": "报告期",
    "report_type": "报告类型",
    "comp_type": "公司类型(1一般工商业2银行3保险4证券)",
    "end_type": "报告期类型",
    "basic_eps": "基本每股收益",
    "diluted_eps": "稀释每股收益",
    "total_revenue": "营业总收入",
    "revenue": "营业收入",
    "int_income": "利息收入",
    "prem_earned": "已赚保费",
    "comm_income": "手续费及佣金收入",
    "n_commis_income": "手续费及佣金净收入",
    "n_oth_income": "其他经营净收益",
    "n_oth_b_income": "加:其他业务净收益",
    "prem_income": "保险业务收入",
    "out_prem": "减:分出保费",
    "une_prem_reser": "提取未到期责任准备金",
    "reins_income": "其中:分保费收入",
    "n_sec_tb_income": "代理买卖证券业务净收入",
    "n_sec_uw_income": "证券承销业务净收入",
    "n_asset_mg_income": "受托客户资产管理业务净收入",
    "oth_b_income": "其他业务收入",
    "fv_value_chg_gain": "加:公允价值变动净收益",
    "invest_income": "加:投资净收益",
    "ass_invest_income": "其中:对联营企业和合营企业的投资收益",
    "forex_gain": "加:汇兑净收益",
    "total_cogs": "营业总成本",
    "oper_cost": "减:营业成本",
    "int_exp": "减:利息支出",
    "comm_exp": "减:手续费及佣金支出",
    "biz_tax_surchg": "减:营业税金及附加",
    "sell_exp": "减:销售费用",
    "admin_exp": "减:管理费用",
    "fin_exp": "减:财务费用",
    "assets_impair_loss": "减:资产减值损失",
    "prem_refund": "退保金",
    "compens_payout": "赔付总支出",
    "reser_insur_liab": "提取保险责任准备金",
    "div_payt": "保户红利支出",
    "reins_exp": "分保费用",
    "oper_exp": "营业支出",
    "compens_payout_refu": "减:摊回赔付支出",
    "insur_reser_refu": "减:摊回保险责任准备金",
    "reins_cost_refund": "减:摊回分保费用",
    "other_bus_cost": "其他业务成本",
    "operate_profit": "营业利润",
    "non_oper_income": "加:营业外收入",
    "non_oper_exp": "减:营业外支出",
    "nca_disploss": "其中:减:非流动资产处置净损失",
    "total_profit": "利润总额",
    "income_tax": "所得税费用",
    "n_income": "净利润(含少数股东损益)",
    "n_income_attr_p": "净利润(不含少数股东损益)",
    "minority_gain": "少数股东损益",
    "oth_compr_income": "其他综合收益",
    "t_compr_income": "综合收益总额",
    "compr_inc_attr_p": "归属于母公司(或股东)的综合收益总额",
    "compr_inc_attr_m_s": "归属于少数股东的综合收益总额",
    "ebit": "息税前利润",
    "ebitda": "息税折旧摊销前利润",
    "insurance_exp": "保险业务支出",
    "undist_profit": "年初未分配利润",
    "distable_profit": "可分配利润",
    "rd_exp": "研发费用",
    "fin_exp_int_exp": "财务费用:利息费用",
    "fin_exp_int_inc": "财务费用:利息收入",
    "transfer_surplus_rese": "盈余公积转入",
    "transfer_housing_imprest": "住房周转金转入",
    "transfer_oth": "其他转入",
    "adj_lossgain": "调整以前年度损益",
    "withdra_legal_surplus": "提取法定盈余公积",
    "withdra_legal_pubfund": "提取法定公益金",
    "withdra_biz_devfund": "提取企业发展基金",
    "withdra_rese_fund": "提取储备基金",
    "withdra_oth_ersu": "提取任意盈余公积金",
    "workers_welfare": "职工奖金福利",
    "distr_profit_shrhder": "可供股东分配的利润",
    "prfshare_payable_dvd": "应付优先股股利",
    "comshare_payable_dvd": "应付普通股股利",
    "capit_comstock_div": "转作股本的普通股股利",
    "net_after_nr_lp_correct": "扣除非经常性损益后的净利润（更正前）",
    "credit_impa_loss": "信用减值损失",
    "net_expo_hedging_benefits": "净敞口套期收益",
    "oth_impair_loss_assets": "其他资产减值损失",
    "total_opcost": "营业总成本（二）",
    "amodcost_fin_assets": "以摊余成本计量的金融资产终止确认收益",
    "oth_income": "其他收益",
    "asset_disp_income": "资产处置收益",
    "continued_net_profit": "持续经营净利润",
    "end_net_profit": "终止经营净利润",
    "update_flag": "更新标识(1最新)"
}

stock_balancesheet_mapping = {
    "ticker": "股票代码",
    "ann_date": "公告日期",
    "f_ann_date": "实际公告日期",
    "end_date": "报告期",
    "report_type": "报表类型",
    "comp_type": "公司类型(1一般工商业2银行3保险4证券)",
    "end_type": "报告期类型",
    "total_share": "期末总股本",
    "cap_rese": "资本公积金",
    "undistr_porfit": "未分配利润",
    "surplus_rese": "盈余公积金",
    "special_rese": "专项储备",
    "money_cap": "货币资金",
    "trad_asset": "交易性金融资产",
    "notes_receiv": "应收票据",
    "accounts_receiv": "应收账款",
    "oth_receiv": "其他应收款",
    "prepayment": "预付款项",
    "div_receiv": "应收股利",
    "int_receiv": "应收利息",
    "inventories": "存货",
    "amor_exp": "待摊费用",
    "nca_within_1y": "一年内到期的非流动资产",
    "sett_rsrv": "结算备付金",
    "loanto_oth_bank_fi": "拆出资金",
    "premium_receiv": "应收保费",
    "reinsur_receiv": "应收分保账款",
    "reinsur_res_receiv": "应收分保合同准备金",
    "pur_resale_fa": "买入返售金融资产",
    "oth_cur_assets": "其他流动资产",
    "total_cur_assets": "流动资产合计",
    "fa_avail_for_sale": "可供出售金融资产",
    "htm_invest": "持有至到期投资",
    "lt_eqt_invest": "长期股权投资",
    "invest_real_estate": "投资性房地产",
    "time_deposits": "定期存款",
    "oth_assets": "其他资产",
    "lt_rec": "长期应收款",
    "fix_assets": "固定资产",
    "cip": "在建工程",
    "const_materials": "工程物资",
    "fixed_assets_disp": "固定资产清理",
    "produc_bio_assets": "生产性生物资产",
    "oil_and_gas_assets": "油气资产",
    "intan_assets": "无形资产",
    "r_and_d": "研发支出",
    "goodwill": "商誉",
    "lt_amor_exp": "长期待摊费用",
    "defer_tax_assets": "递延所得税资产",
    "decr_in_disbur": "发放贷款及垫款",
    "oth_nca": "其他非流动资产",
    "total_nca": "非流动资产合计",
    "cash_reser_cb": "现金及存放中央银行款项",
    "depos_in_oth_bfi": "存放同业和其它金融机构款项",
    "prec_metals": "贵金属",
    "deriv_assets": "衍生金融资产",
    "rr_reins_une_prem": "应收分保未到期责任准备金",
    "rr_reins_outstd_cla": "应收分保未决赔款准备金",
    "rr_reins_lins_liab": "应收分保寿险责任准备金",
    "rr_reins_lthins_liab": "应收分保长期健康险责任准备金",
    "refund_depos": "存出保证金",
    "ph_pledge_loans": "保户质押贷款",
    "refund_cap_depos": "存出资本保证金",
    "indep_acct_assets": "独立账户资产",
    "client_depos": "其中：客户资金存款",
    "client_prov": "其中：客户备付金",
    "transac_seat_fee": "其中:交易席位费",
    "invest_as_receiv": "应收款项类投资",
    "total_assets": "资产总计",
    "lt_borr": "长期借款",
    "st_borr": "短期借款",
    "cb_borr": "向中央银行借款",
    "depos_ib_deposits": "吸收存款及同业存放",
    "loan_oth_bank": "拆入资金",
    "trading_fl": "交易性金融负债",
    "notes_payable": "应付票据",
    "acct_payable": "应付账款",
    "adv_receipts": "预收款项",
    "sold_for_repur_fa": "卖出回购金融资产款",
    "comm_payable": "应付手续费及佣金",
    "payroll_payable": "应付职工薪酬",
    "taxes_payable": "应交税费",
    "int_payable": "应付利息",
    "div_payable": "应付股利",
    "oth_payable": "其他应付款",
    "acc_exp": "预提费用",
    "deferred_inc": "递延收益",
    "st_bonds_payable": "应付短期债券",
    "payable_to_reinsurer": "应付分保账款",
    "rsrv_insur_cont": "保险合同准备金",
    "acting_trading_sec": "代理买卖证券款",
    "acting_uw_sec": "代理承销证券款",
    "non_cur_liab_due_1y": "一年内到期的非流动负债",
    "oth_cur_liab": "其他流动负债",
    "total_cur_liab": "流动负债合计",
    "bond_payable": "应付债券",
    "lt_payable": "长期应付款",
    "specific_payables": "专项应付款",
    "estimated_liab": "预计负债",
    "defer_tax_liab": "递延所得税负债",
    "defer_inc_non_cur_liab": "递延收益-非流动负债",
    "oth_ncl": "其他非流动负债",
    "total_ncl": "非流动负债合计",
    "depos_oth_bfi": "同业和其它金融机构存放款项",
    "deriv_liab": "衍生金融负债",
    "depos": "吸收存款",
    "agency_bus_liab": "代理业务负债",
    "oth_liab": "其他负债",
    "prem_receiv_adva": "预收保费",
    "depos_received": "存入保证金",
    "ph_invest": "保户储金及投资款",
    "reser_une_prem": "未到期责任准备金",
    "reser_outstd_claims": "未决赔款准备金",
    "reser_lins_liab": "寿险责任准备金",
    "reser_lthins_liab": "长期健康险责任准备金",
    "indept_acc_liab": "独立账户负债",
    "pledge_borr": "其中:质押借款",
    "indem_payable": "应付赔付款",
    "policy_div_payable": "应付保单红利",
    "total_liab": "负债合计",
    "treasury_share": "减:库存股",
    "ordin_risk_reser": "一般风险准备",
    "forex_differ": "外币报表折算差额",
    "invest_loss_unconf": "未确认的投资损失",
    "minority_int": "少数股东权益",
    "total_hldr_eqy_exc_min_int": "股东权益合计(不含少数股东权益)",
    "total_hldr_eqy_inc_min_int": "股东权益合计(含少数股东权益)",
    "total_liab_hldr_eqy": "负债及股东权益总计",
    "lt_payroll_payable": "长期应付职工薪酬",
    "oth_comp_income": "其他综合收益",
    "oth_eqt_tools": "其他权益工具",
    "oth_eqt_tools_p_shr": "其他权益工具(优先股)",
    "lending_funds": "融出资金",
    "acc_receivable": "应收款项",
    "st_fin_payable": "应付短期融资款",
    "payables": "应付款项",
    "hfs_assets": "持有待售的资产",
    "hfs_sales": "持有待售的负债",
    "cost_fin_assets": "以摊余成本计量的金融资产",
    "fair_value_fin_assets": "以公允价值计量且其变动计入其他综合收益的金融资产",
    "cip_total": "在建工程(合计)(元)",
    "oth_pay_total": "其他应付款(合计)(元)",
    "long_pay_total": "长期应付款(合计)(元)",
    "debt_invest": "债权投资(元)",
    "oth_debt_invest": "其他债权投资(元)",
    "oth_eq_invest": "其他权益工具投资(元)",
    "oth_illiq_fin_assets": "其他非流动金融资产(元)",
    "oth_eq_ppbond": "其他权益工具:永续债(元)",
    "receiv_financing": "应收款项融资",
    "use_right_assets": "使用权资产",
    "lease_liab": "租赁负债",
    "contract_assets": "合同资产",
    "contract_liab": "合同负债",
    "accounts_receiv_bill": "应收票据及应收账款",
    "accounts_pay": "应付票据及应付账款",
    "oth_rcv_total": "其他应收款(合计)（元）",
    "fix_assets_total": "固定资产(合计)(元)",
    "update_flag": "更新标识(1最新)"
}

stock_cashflow_mapping = {
    "ticker": "股票代码",
    "ann_date": "公告日期",
    "f_ann_date": "实际公告日期",
    "end_date": "报告期",
    "comp_type": "公司类型(1一般工商业2银行3保险4证券)",
    "report_type": "报表类型",
    "end_type": "报告期类型",
    "net_profit": "净利润",
    "finan_exp": "财务费用",
    "c_fr_sale_sg": "销售商品、提供劳务收到的现金",
    "recp_tax_rends": "收到的税费返还",
    "n_depos_incr_fi": "客户存款和同业存放款项净增加额",
    "n_incr_loans_cb": "向中央银行借款净增加额",
    "n_inc_borr_oth_fi": "向其他金融机构拆入资金净增加额",
    "prem_fr_orig_contr": "收到原保险合同保费取得的现金",
    "n_incr_insured_dep": "保户储金净增加额",
    "n_reinsur_prem": "收到再保业务现金净额",
    "n_incr_disp_tfa": "处置交易性金融资产净增加额",
    "ifc_cash_incr": "收取利息和手续费净增加额",
    "n_incr_disp_faas": "处置可供出售金融资产净增加额",
    "n_incr_loans_oth_bank": "拆入资金净增加额",
    "n_cap_incr_repur": "回购业务资金净增加额",
    "c_fr_oth_operate_a": "收到其他与经营活动有关的现金",
    "c_inf_fr_operate_a": "经营活动现金流入小计",
    "c_paid_goods_s": "购买商品、接受劳务支付的现金",
    "c_paid_to_for_empl": "支付给职工以及为职工支付的现金",
    "c_paid_for_taxes": "支付的各项税费",
    "n_incr_clt_loan_adv": "客户贷款及垫款净增加额",
    "n_incr_dep_cbob": "存放央行和同业款项净增加额",
    "c_pay_claims_orig_inco": "支付原保险合同赔付款项的现金",
    "pay_handling_chrg": "支付手续费的现金",
    "pay_comm_insur_plcy": "支付保单红利的现金",
    "oth_cash_pay_oper_act": "支付其他与经营活动有关的现金",
    "st_cash_out_act": "经营活动现金流出小计",
    "n_cashflow_act": "经营活动产生的现金流量净额",
    "oth_recp_ral_inv_act": "收到其他与投资活动有关的现金",
    "c_disp_withdrwl_invest": "收回投资收到的现金",
    "c_recp_return_invest": "取得投资收益收到的现金",
    "n_recp_disp_fiolta": "处置固定资产、无形资产和其他长期资产收回的现金净额",
    "n_recp_disp_sobu": "处置子公司及其他营业单位收到的现金净额",
    "stot_inflows_inv_act": "投资活动现金流入小计",
    "c_pay_acq_const_fiolta": "购建固定资产、无形资产和其他长期资产支付的现金",
    "c_paid_invest": "投资支付的现金",
    "n_disp_subs_oth_biz": "取得子公司及其他营业单位支付的现金净额",
    "oth_pay_ral_inv_act": "支付其他与投资活动有关的现金",
    "n_incr_pledge_loan": "质押贷款净增加额",
    "stot_out_inv_act": "投资活动现金流出小计",
    "n_cashflow_inv_act": "投资活动产生的现金流量净额",
    "c_recp_borrow": "取得借款收到的现金",
    "proc_issue_bonds": "发行债券收到的现金",
    "oth_cash_recp_ral_fnc_act": "收到其他与筹资活动有关的现金",
    "stot_cash_in_fnc_act": "筹资活动现金流入小计",
    "free_cashflow": "企业自由现金流量",
    "c_prepay_amt_borr": "偿还债务支付的现金",
    "c_pay_dist_dpcp_int_exp": "分配股利、利润或偿付利息支付的现金",
    "incl_dvd_profit_paid_sc_ms": "其中:子公司支付给少数股东的股利、利润",
    "oth_cashpay_ral_fnc_act": "支付其他与筹资活动有关的现金",
    "stot_cashout_fnc_act": "筹资活动现金流出小计",
    "n_cash_flows_fnc_act": "筹资活动产生的现金流量净额",
    "eff_fx_flu_cash": "汇率变动对现金的影响",
    "n_incr_cash_cash_equ": "现金及现金等价物净增加额",
    "c_cash_equ_beg_period": "期初现金及现金等价物余额",
    "c_cash_equ_end_period": "期末现金及现金等价物余额",
    "c_recp_cap_contrib": "吸收投资收到的现金",
    "incl_cash_rec_saims": "其中:子公司吸收少数股东投资收到的现金",
    "uncon_invest_loss": "未确认投资损失",
    "prov_depr_assets": "加:资产减值准备",
    "depr_fa_coga_dpba": "固定资产折旧、油气资产折耗、生产性生物资产折旧",
    "amort_intang_assets": "无形资产摊销",
    "lt_amort_deferred_exp": "长期待摊费用摊销",
    "decr_deferred_exp": "待摊费用减少",
    "incr_acc_exp": "预提费用增加",
    "loss_disp_fiolta": "处置固定、无形资产和其他长期资产的损失",
    "loss_scr_fa": "固定资产报废损失",
    "loss_fv_chg": "公允价值变动损失",
    "invest_loss": "投资损失",
    "decr_def_inc_tax_assets": "递延所得税资产减少",
    "incr_def_inc_tax_liab": "递延所得税负债增加",
    "decr_inventories": "存货的减少",
    "decr_oper_payable": "经营性应收项目的减少",
    "incr_oper_payable": "经营性应付项目的增加",
    "others": "其他",
    "im_net_cashflow_oper_act": "经营活动产生的现金流量净额(间接法)",
    "conv_debt_into_cap": "债务转为资本",
    "conv_copbonds_due_within_1y": "一年内到期的可转换公司债券",
    "fa_fnc_leases": "融资租入固定资产",
    "im_n_incr_cash_equ": "现金及现金等价物净增加额(间接法)",
    "net_dism_capital_add": "拆出资金净增加额",
    "net_cash_rece_sec": "代理买卖证券收到的现金净额(元)",
    "credit_impa_loss": "信用减值损失",
    "use_right_asset_dep": "使用权资产折旧",
    "oth_loss_asset": "其他资产减值损失",
    "end_bal_cash": "现金的期末余额",
    "beg_bal_cash": "减:现金的期初余额",
    "end_bal_cash_equ": "加:现金等价物的期末余额",
    "beg_bal_cash_equ": "减:现金等价物的期初余额",
    "update_flag": "更新标志(1最新)"
}

financial_report_mapping = {
    "stock_income": stock_income_mapping,
    "stock_balancesheet": stock_balancesheet_mapping,
    "stock_cashflow": stock_cashflow_mapping
}