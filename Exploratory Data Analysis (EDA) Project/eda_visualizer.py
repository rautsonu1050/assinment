# ============================================================
# VISUALIZATION MODULE
# ============================================================

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from eda_config import CURRENT_DIR, FIGURE_DPI, COLORS, SURVIVAL_COLORS, CLASS_COLORS

# Set style for better visualizations
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)

def create_dashboard(df):
    """Create comprehensive visualization dashboard"""
    print("\n" + "="*70)
    print("GENERATING VISUALIZATION DASHBOARD")
    print("="*70)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    # Create figure with subplots
    fig = plt.figure(figsize=(22, 18))
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)
    
    # 1. Age Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    if 'age' in df.columns:
        sns.histplot(df['age'].dropna(), bins=30, kde=True, color=COLORS['primary'], alpha=0.7)
        ax1.set_title('Age Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Age')
        ax1.set_ylabel('Frequency')
        ax1.axvline(df['age'].mean(), color=COLORS['secondary'], linestyle='--', 
                   linewidth=2, label=f'Mean: {df["age"].mean():.1f}')
        ax1.axvline(df['age'].median(), color=COLORS['success'], linestyle='-.', 
                   linewidth=2, label=f'Median: {df["age"].median():.1f}')
        ax1.legend()
    
    # 2. Fare Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    if 'fare' in df.columns:
        sns.histplot(df['fare'].dropna(), bins=30, kde=True, color=COLORS['secondary'], alpha=0.7)
        ax2.set_title('Fare Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Fare')
        ax2.set_ylabel('Frequency')
        ax2.axvline(df['fare'].mean(), color=COLORS['primary'], linestyle='--', 
                   linewidth=2, label=f'Mean: {df["fare"].mean():.1f}')
        ax2.axvline(df['fare'].median(), color=COLORS['success'], linestyle='-.', 
                   linewidth=2, label=f'Median: {df["fare"].median():.1f}')
        ax2.legend()
    
    # 3. Class Distribution
    ax3 = fig.add_subplot(gs[0, 2])
    if 'pclass' in df.columns:
        class_counts = df['pclass'].value_counts().sort_index()
        wedges, texts, autotexts = ax3.pie(class_counts.values, 
                                           labels=[f'Class {i}' for i in class_counts.index], 
                                           autopct='%1.1f%%', 
                                           colors=CLASS_COLORS, 
                                           startangle=90,
                                           explode=(0.05, 0.02, 0.02))
        ax3.set_title('Passenger Class Distribution', fontsize=14, fontweight='bold')
    
    # 4. Survival by Gender
    ax4 = fig.add_subplot(gs[1, 0])
    if 'survived' in df.columns and 'sex' in df.columns:
        gender_survival = df.groupby('sex')['survived'].mean() * 100
        bars = ax4.bar(gender_survival.index, gender_survival.values, 
                      color=[COLORS['primary'], COLORS['secondary']])
        ax4.set_title('Survival Rate by Gender', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Survival Rate (%)')
        ax4.set_ylim(0, 100)
        for bar, value in zip(bars, gender_survival.values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 5. Survival by Class
    ax5 = fig.add_subplot(gs[1, 1])
    if 'survived' in df.columns and 'pclass' in df.columns:
        class_survival = df.groupby('pclass')['survived'].mean() * 100
        bars = ax5.bar(class_survival.index, class_survival.values, color=CLASS_COLORS)
        ax5.set_title('Survival Rate by Passenger Class', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Passenger Class')
        ax5.set_ylabel('Survival Rate (%)')
        ax5.set_ylim(0, 100)
        for bar, value in zip(bars, class_survival.values):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 6. Correlation Heatmap
    ax6 = fig.add_subplot(gs[1, 2])
    if len(numerical_cols) > 1:
        corr = df[numerical_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', center=0, 
                   fmt='.2f', square=True, linewidths=0.5, ax=ax6, 
                   cbar_kws={"shrink": 0.8})
        ax6.set_title('Correlation Heatmap', fontsize=14, fontweight='bold')
    
    # 7. Age by Class
    ax7 = fig.add_subplot(gs[2, 0])
    if 'age' in df.columns and 'pclass' in df.columns:
        sns.boxplot(data=df, x='pclass', y='age', palette='Set3')
        ax7.set_title('Age Distribution by Class', fontsize=14, fontweight='bold')
        ax7.set_xlabel('Passenger Class')
        ax7.set_ylabel('Age')
    
    # 8. Fare by Class
    ax8 = fig.add_subplot(gs[2, 1])
    if 'fare' in df.columns and 'pclass' in df.columns:
        sns.boxplot(data=df, x='pclass', y='fare', palette='Set2')
        ax8.set_title('Fare Distribution by Class', fontsize=14, fontweight='bold')
        ax8.set_xlabel('Passenger Class')
        ax8.set_ylabel('Fare')
    
    # 9. Age vs Fare Scatter
    ax9 = fig.add_subplot(gs[2, 2])
    if 'age' in df.columns and 'fare' in df.columns:
        if 'survived' in df.columns:
            scatter = sns.scatterplot(data=df, x='age', y='fare', 
                                    hue='survived', alpha=0.6, 
                                    palette=SURVIVAL_COLORS, s=50)
            ax9.set_title('Age vs Fare by Survival', fontsize=14, fontweight='bold')
            legend = ax9.legend(title='Survived', labels=['No', 'Yes'])
        else:
            sns.scatterplot(data=df, x='age', y='fare', alpha=0.6, color=COLORS['primary'], s=50)
            ax9.set_title('Age vs Fare', fontsize=14, fontweight='bold')
        ax9.set_xlabel('Age')
        ax9.set_ylabel('Fare')
    
    # 10. Survival by Gender and Class
    ax10 = fig.add_subplot(gs[3, 0])
    if 'survived' in df.columns and 'sex' in df.columns and 'pclass' in df.columns:
        survival_pivot = df.groupby(['sex', 'pclass'])['survived'].mean().unstack() * 100
        survival_pivot.plot(kind='bar', ax=ax10, color=CLASS_COLORS, edgecolor='black')
        ax10.set_title('Survival Rate: Gender × Class', fontsize=14, fontweight='bold')
        ax10.set_ylabel('Survival Rate (%)')
        ax10.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax10.set_xlabel('Gender')
        ax10.set_ylim(0, 100)
        for container in ax10.containers:
            ax10.bar_label(container, fmt='%.1f%%', fontsize=9)
    
    # 11. Age Distribution by Survival
    ax11 = fig.add_subplot(gs[3, 1])
    if 'survived' in df.columns and 'age' in df.columns:
        sns.kdeplot(data=df, x='age', hue='survived', fill=True, 
                   common_norm=False, alpha=0.4, palette=SURVIVAL_COLORS, linewidth=2)
        ax11.set_title('Age Distribution by Survival', fontsize=14, fontweight='bold')
        ax11.set_xlabel('Age')
        ax11.set_ylabel('Density')
        legend = ax11.legend(title='Survived', labels=['No', 'Yes'])
    
    # 12. Fare Distribution by Survival
    ax12 = fig.add_subplot(gs[3, 2])
    if 'survived' in df.columns and 'fare' in df.columns:
        sns.kdeplot(data=df, x='fare', hue='survived', fill=True,
                   common_norm=False, alpha=0.4, palette=SURVIVAL_COLORS, linewidth=2)
        ax12.set_title('Fare Distribution by Survival', fontsize=14, fontweight='bold')
        ax12.set_xlabel('Fare')
        ax12.set_ylabel('Density')
        legend = ax12.legend(title='Survived', labels=['No', 'Yes'])
    
    plt.suptitle('EXPLORATORY DATA ANALYSIS DASHBOARD', fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    save_path = f"{CURRENT_DIR}/eda_dashboard.png"
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white')
    print(f"✅ Dashboard saved: {save_path}")
    
    plt.show()
    plt.close()

def create_correlation_analysis(df):
    """Create detailed correlation analysis"""
    print("\n" + "="*70)
    print("GENERATING CORRELATION ANALYSIS")
    print("="*70)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numerical_cols) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Full correlation matrix
        corr = df[numerical_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', center=0, 
                   fmt='.2f', square=True, linewidths=0.5, ax=axes[0],
                   cbar_kws={"shrink": 0.8})
        axes[0].set_title('Complete Correlation Matrix', fontsize=14, fontweight='bold')
        
        # Correlation with survival
        if 'survived' in df.columns:
            target_corr = corr['survived'].drop('survived').sort_values()
            colors = [COLORS['secondary'] if x < 0 else COLORS['success'] for x in target_corr]
            bars = axes[1].barh(target_corr.index, target_corr.values, color=colors, edgecolor='black')
            axes[1].set_title('Correlation with Survival', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Correlation Coefficient')
            axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            axes[1].set_xlim(-1, 1)
            for i, v in enumerate(target_corr.values):
                axes[1].text(v + 0.02, i, f'{v:.3f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        save_path = f"{CURRENT_DIR}/correlation_analysis.png"
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white')
        print(f"✅ Correlation analysis saved: {save_path}")
        plt.show()
        plt.close()

def create_pairplot(df):
    """Create pairplot for key variables"""
    print("\n" + "="*70)
    print("GENERATING PAIRPLOT")
    print("="*70)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numerical_cols) >= 2:
        key_cols = ['age', 'fare', 'pclass']
        if 'survived' in df.columns:
            key_cols.append('survived')
        
        available_cols = [col for col in key_cols if col in df.columns]
        if len(available_cols) >= 2:
            g = sns.pairplot(df[available_cols], 
                           hue='survived' if 'survived' in df.columns else None,
                           palette=SURVIVAL_COLORS if 'survived' in df.columns else 'viridis',
                           diag_kind='kde',
                           diag_kws={'fill': True, 'alpha': 0.5},
                           plot_kws={'alpha': 0.6, 's': 40},
                           height=2.5)
            g.fig.suptitle('Pairwise Relationships', fontsize=16, fontweight='bold', y=1.02)
            if 'survived' in df.columns:
                g._legend.set_title('Survived')
                for text in g._legend.texts:
                    if text.get_text() == '0':
                        text.set_text('No')
                    elif text.get_text() == '1':
                        text.set_text('Yes')
            save_path = f"{CURRENT_DIR}/pairplot.png"
            plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white')
            print(f"✅ Pairplot saved: {save_path}")
            plt.show()
            plt.close()